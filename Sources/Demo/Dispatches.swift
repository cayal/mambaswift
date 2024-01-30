import Foundation
import Metal

enum MambaRunnerError: Error {
    case invalidParameterShape(String),
         missingData(String),
         failedToMakeMetalBuffer,
         failedToMakeCommandBuffer,
         failedToMakeCommandEncoder,
         failedToMakeArgumentEncoder,
         shapeMismatch([UInt32], [UInt32]),
         missingFunction(String),
         sanityMissing
}

class MambaRunnerB {
    public var hp: OGHParams
    public var state: MBLState<OG130Spec>
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    
    public let smokeTestPS: MTLComputePipelineState
    public let embedPS: MTLComputePipelineState
    public let rmsNormAndCopyPS: MTLComputePipelineState
    public let rowSquaredSumsPS: MTLComputePipelineState
    public let inProjPS: MTLComputePipelineState
    public let conv1d4dconv4PS: MTLComputePipelineState
    public let xProjToDBCPS: MTLComputePipelineState
    public let softplusPS: MTLComputePipelineState
    public let dtProjPS: MTLComputePipelineState
    public let discretizePS: MTLComputePipelineState
    public let dasSSMPS: MTLComputePipelineState
    public let ySiluResPS: MTLComputePipelineState
    public let outProjPS: MTLComputePipelineState
    
    public var maxSSMWidth: Int
    {
        let m = device.maxThreadgroupMemoryLength
        let b = MemoryLayout<Float32>.stride
        let n = Int(hp.dState)
        let w = ((m/b) - 2*n) / (5*n + 3)
        let cap = device.maxThreadsPerThreadgroup.width / Int(hp.dState)
        return max( min( w, cap ), cap)
    }
    
    public init(modelSpec: OG130Spec, state: MBLState<OG130Spec>, device: MTLDevice, cmdQ: MTLCommandQueue, library: MTLLibrary) throws {
        self.hp = modelSpec.hp
        self.state = state
        self.commandQueue = cmdQ
        self.device = device
        self.library = library
        
        let mtlfn: (String) throws -> MTLFunction = {
            guard let x = library.makeFunction(name: $0) else {throw MambaRunnerError.missingFunction($0)}
            return x
        }
        
        let smokeTest   = try mtlfn("smokeTest")
        let embed       = try mtlfn("getEmbeddings")
        let rmsNormAndCopy     = try mtlfn("rmsNormAndCopy")
        let rowSquaredSums      = try mtlfn("rowSquaredSums")
        let inProj = try mtlfn("inProj")
        let conv1d4dconv4 = try mtlfn("conv1d4dconv4")
        let xProjToDBC = try mtlfn("xProjToDBC")
        let softplus = try mtlfn("softplus")
        let dtProj = try mtlfn("dtProj")
        let discretize = try mtlfn("discretize")
        let dasSSM = try mtlfn("discretizeAndScanSSM")
        let ySiluRes = try mtlfn("ySiluRes")
        let outProj = try mtlfn("outProj")
        
        self.smokeTestPS            = try device.makeComputePipelineState(function: smokeTest)
        self.embedPS                = try device.makeComputePipelineState(function: embed)
        self.rmsNormAndCopyPS       = try device.makeComputePipelineState(function: rmsNormAndCopy)
        self.rowSquaredSumsPS       = try device.makeComputePipelineState(function: rowSquaredSums)
        self.inProjPS               = try device.makeComputePipelineState(function: inProj)
        self.conv1d4dconv4PS        = try device.makeComputePipelineState(function: conv1d4dconv4)
        self.xProjToDBCPS           = try device.makeComputePipelineState(function: xProjToDBC)
        self.softplusPS             = try device.makeComputePipelineState(function: softplus)
        self.dtProjPS               = try device.makeComputePipelineState(function: dtProj)
        self.discretizePS           = try device.makeComputePipelineState(function: discretize)
        self.dasSSMPS               = try device.makeComputePipelineState(function: dasSSM)
        self.ySiluResPS             = try device.makeComputePipelineState(function: ySiluRes)
        self.outProjPS              = try device.makeComputePipelineState(function: outProj)
    }
    
    public func scalarBuffer<TNum: Numeric>(_ scalar: TNum) throws -> MTLBuffer {
        var buf: MTLBuffer?
        try withUnsafeBytes(of: scalar) { sp in
            guard let ba = sp.baseAddress else {
                throw MambaRunnerError.invalidParameterShape("Scalar has no base address!?")
            }
            buf = self.device.makeBuffer(bytes: ba, length: MemoryLayout<TNum>.stride)
        }
        guard let someB = buf else {
            throw MambaRunnerError.failedToMakeMetalBuffer
        }
        return someB
    }
    
    public func vectorBuffer<TNum: Numeric>(_ seq: any Sequence<TNum>) throws -> MTLBuffer {
        var arr = ContiguousArray(seq)
        let byteLength = MemoryLayout<TNum>.stride * arr.count
        return try arr.withUnsafeMutableBufferPointer { bp in
            guard let ba = bp.baseAddress else {
                throw MambaRunnerError.invalidParameterShape("Cannot make buffer from empty array")
            }
            guard let buf = self.device.makeBuffer(bytes: ba, length: byteLength) else {
                throw MambaRunnerError.failedToMakeMetalBuffer
            }
            return buf
        }
    }
    
    func setHParams(computeEncoder: inout MTLComputeCommandEncoder, index: Int) {
        withUnsafeMutableBytes(of: &self.hp) { bytes in
            computeEncoder.setBytes(bytes.baseAddress!, length: MemoryLayout<OGHParams>.stride, index: index)
        }
    }
    
    
    private func gcd(_ a: Int, _ b: Int) -> Int {
        let r = a % b
        if r != 0 { return gcd(b, r) }
        return b
    }
    
    /// Return values:
    /// Left/0: Threads per group
    /// Right/1: Threadgroups per grid
    func matchStride(_ width: Int) -> (Int, Int) {
        let maxWidth = device.maxThreadsPerThreadgroup.width
        if (width < maxWidth) { return (width, 1) }
        let gcd = gcd(width, maxWidth)
        return (gcd, width / gcd)
    }
    
    func printAbSmokeTest(x: MBLHeapedParameter) throws {
//        let kernelFunction = library.makeFunction(name: "abSmokeTest")!
//        let computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
//
//        let data = x.data
//
//        let dataSize = data.length / MemoryLayout<Float16>.stride
//        let (tpg, ntg) = matchStride(Int(state.dModel))
//        let threadsPerGroup = MTLSize(width: tpg, height: 1, depth: 1)
//        let numThreadgroups = MTLSize(width: ntg, height: 1, depth: 1)
//
//        let commandBuffer = commandQueue.makeCommandBuffer()!
//        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
//        computeEncoder.setComputePipelineState(computePipelineState)
//        computeEncoder.setBuffer(data, offset: 0, index: 0)
//        try setStateArgBuffers(device: device, computeEncoder: computeEncoder, kernelFunction: kernelFunction, paramsIndex: 1, stateIndex: 2)
//
//        computeEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
//        computeEncoder.endEncoding()
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        var output = [Float16]()
//        let dataPointer = x.contents().bindMemory(to: Float16.self, capacity: dataSize)
//        let bufferPointer = UnsafeBufferPointer(start: dataPointer, count: dataSize)
//        bufferPointer.forEach { value in output.append(value) }
//        print(output)
//        print("Yarg")
    }
    
    func smokeTest(_ ctx: inout OGScratchContext<OG130Spec>) throws {
        guard let n1 = ctx.n1 else { throw MambaRunnerError.missingData("n1") }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "smokeTest"
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        computeEncoder.setComputePipelineState(self.smokeTestPS)
        computeEncoder.useHeaps([ctx.heap, state.heap])
        computeEncoder.useResource(n1.data, usage: .write)
        computeEncoder.setBuffer(ctx.argBuf,        offset: 0,                      index: 0)
        computeEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
        computeEncoder.setBuffer(state.stateArgBuffer,        offset: 0,            index: 2)

        let tpg = MTLSize(width: 255, height: 1, depth: 1)
        let tptg = MTLSize(width: 255, height: 1, depth: 1)
        computeEncoder.dispatchThreads(tpg, threadsPerThreadgroup: tptg)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        print("Yarg")
    }
    
    func simdStripes(origSize inSize: Int, warpSize: Int, max: Int) throws -> (outSize: Int, count: Int) {
        guard inSize <= warpSize || inSize % warpSize == 0 else {
            throw MambaRunnerError.invalidParameterShape("Warp size (\(warpSize) should evenly divide row: (\(inSize)).")
        }
        if 0 <= inSize, inSize < max {
            return (outSize: inSize, count: 1)
        } else {
            let width = gcd(max, inSize)
            return (outSize: width, count: inSize / width)
        }
    }
    
    func tgMaxGroups(totalLengthToDispatch: Int, max: Int = 1024) -> [(dispatchLength: Int, offset: Int)] {
        let remainder = totalLengthToDispatch % max
        let blocks = totalLengthToDispatch / max
        return Array(0..<blocks).map { (dispatchLength: max      , offset: $0 * max) }
                                    + [(dispatchLength: remainder, offset: blocks * max)]
    }
    
    func tgBoxForRow(_ rowSize: Int, comps: MTLComputePipelineState) throws -> MTLSize {
        let ws =  comps.threadExecutionWidth
        let max = comps.maxTotalThreadsPerThreadgroup
        guard rowSize % ws == 0 else {
            throw MambaRunnerError.invalidParameterShape("Warp size (\(ws) should evenly divide row: (\(rowSize)).")
        }
        switch rowSize {
        case 0...max:
            return MTLSize(width: rowSize, height: 1, depth: 1)
        case (max + 1)..<(max * gcd(max, rowSize)):
            let warpCount = (rowSize + ws - 1) / ws
            let width = gcd(max, rowSize)
            let height = rowSize / width
            return MTLSize(width: width, height: height, depth: 1)
        default:
            throw MambaRunnerError.invalidParameterShape("\(rowSize) / \(gcd(max, rowSize)) > \(max)")
        }
    }
    
    public func inProj(_ ctx: inout OGScratchContext<OG130Spec>, ln: inout UInt32) throws -> MTLBuffer {
        guard let dbldIn = ctx.dbld_in else { throw MambaRunnerError.missingData("dbldIn") }
        guard let inProjW = state.layers[Int(ln)].in_proj_weight else { throw MambaRunnerError.missingData("layer\(ln) in_proj_weight") }
        let rowWidthIn  = Int(hp.dModel)
        var rowWidthOut = UInt32(ctx.L)
        let rowCountOut = Int(hp.dInProj)
        let inProjPS = self.inProjPS
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "inProj@\(ln)"
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        computeEncoder.setComputePipelineState(inProjPS)
        computeEncoder.useHeaps([ctx.heap, state.heap])
        computeEncoder.useResource(dbldIn.data, usage: .write)
        computeEncoder.setBuffer(ctx.argBuf,        offset: 0,                      index: 0)
        computeEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
        computeEncoder.useResource(inProjW.data, usage: .read)
        computeEncoder.setBuffer(state.stateArgBuffer,        offset: 0,            index: 2)
        
        let (tgWidth, tgsPerRowIn)   = try simdStripes(origSize: rowWidthIn, warpSize: conv1d4dconv4PS.threadExecutionWidth, max: conv1d4dconv4PS.maxTotalThreadsPerThreadgroup)
        let threadsPerThreadgroup    = MTLSize(width: tgWidth, height: 1, depth: 1)
        
        let threadgroupsPerOutVal = tgsPerRowIn * Int(rowWidthOut)
        
        let zeds = try vectorBuffer(Array<Float32>.init(repeating: 0.0, count: Int(rowWidthOut) * rowCountOut))
        
        computeEncoder.setBuffer(zeds,        offset: 0,                            index: 3)
        computeEncoder.setBytes(&ln,        length: MemoryLayout<UInt32>.stride,    index: 4)
        computeEncoder.setBytes(&rowWidthOut, length: MemoryLayout<UInt32>.stride,  index: 5)
                
        let threadgroupsPerGrid = MTLSize(width: threadgroupsPerOutVal, height: rowCountOut, depth: 1)
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return dbldIn.data
    }
    
    public func conv1d4dconv4(_ ctx: inout OGScratchContext<OG130Spec>, ln: inout UInt32) throws {
        guard hp.dConv == 4 else { throw MambaRunnerError.invalidParameterShape("conv1d4dconv4 can only conv1d 4 dconv of 4") }
        guard let dbldIn = ctx.dbld_in else { throw MambaRunnerError.missingData("dbldIn") }
        guard let deli = ctx.deli else { throw MambaRunnerError.missingData("deli") }
        guard let conv1dW = state.layers[Int(ln)].conv1d_weight else { throw MambaRunnerError.missingData("layer\(ln) in_proj_weight") }
        guard let conv1dB = state.layers[Int(ln)].conv1d_bias else { throw MambaRunnerError.missingData("layer\(ln) in_proj_weight") }
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "conv1d4dconv4@\(ln)"
        
        // Blit pass from [0, dInner] of dbld_in to deli to buffer convolution pass
        guard let deliBlit = commandBuffer.makeBlitCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        deliBlit.copy(from: dbldIn.data, sourceOffset: 0, to: deli.data, destinationOffset: 0, size: Int(ctx.L)*Int(hp.dInner) * MemoryLayout<Float32>.stride);
        deliBlit.endEncoding()
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        let conv1d4dconv4PS = self.conv1d4dconv4PS
        computeEncoder.setComputePipelineState(conv1d4dconv4PS)
        
        let colGroups = try simdStripes(origSize: Int(hp.dInner),
                                        warpSize: conv1d4dconv4PS.threadExecutionWidth,
                                        max: conv1d4dconv4PS.maxTotalThreadsPerThreadgroup / 4)
        let threadsPerOutX = 4;
        var xWidthOut = UInt32(ctx.L);
        
        computeEncoder.useHeaps([ctx.heap, state.heap])
        computeEncoder.useResource(dbldIn.data, usage: .write)
        computeEncoder.setBuffer(ctx.argBuf,        offset: 0,                      index: 0)
        computeEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
        computeEncoder.useResource(conv1dW.data, usage: .read)
        computeEncoder.useResource(conv1dB.data, usage: .read)
        computeEncoder.useResource(deli.data, usage: .read)
        computeEncoder.setBuffer(state.stateArgBuffer,        offset: 0,            index: 2)
        computeEncoder.setBytes(&ln,        length: MemoryLayout<UInt32>.stride,    index: 3)
        computeEncoder.setBytes(&xWidthOut, length: MemoryLayout<UInt32>.stride,  index: 4)
        
        let threadsPerThreadgroup = MTLSize(width: threadsPerOutX, height: colGroups.outSize, depth: 1)
        
        // dConv of 4
        let tgpg = MTLSize(width: ctx.L, height: colGroups.count, depth: 1)
        computeEncoder.dispatchThreadgroups(tgpg, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    public func blitsAndChips(_ ctx: inout OGScratchContext<OG130Spec>) throws {
        guard let ld2 = ctx.ld2 else { throw MambaRunnerError.missingData("ld2") }
        guard let dA_l = ctx.dA_l else { throw MambaRunnerError.missingData("dA_l") }
        guard let dBu_l = ctx.dBu_l else { throw MambaRunnerError.missingData("dBu_l") }
        guard let dbld_in = ctx.dbld_in else { throw MambaRunnerError.missingData("dbld_in") }
        guard let debici = ctx.debici else { throw MambaRunnerError.missingData("debici") }
        guard let deli = ctx.deli else { throw MambaRunnerError.missingData("deli") }
        guard let tricky = ctx.tricky else { throw MambaRunnerError.missingData("tricky") }
        guard let n1 = ctx.n1 else { throw MambaRunnerError.missingData("n1") }
        guard let ssx = ctx.ssx else { throw MambaRunnerError.missingData("ssx") }

        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "blitsAndChips"
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        blitEncoder.__fill(ld2.data, range: .init(location: 0, length: ld2.data.length), value: 0)
        blitEncoder.__fill(dA_l.data, range: .init(location: 0, length: dA_l.data.length), value: 0)
        blitEncoder.__fill(dBu_l.data, range: .init(location: 0, length: dBu_l.data.length), value: 0)
        blitEncoder.__fill(dbld_in.data, range: .init(location: 0, length: dbld_in.data.length), value: 0)
        blitEncoder.__fill(debici.data, range: .init(location: 0, length: debici.data.length), value: 0)
        blitEncoder.__fill(deli.data, range: .init(location: 0, length: deli.data.length), value: 0)
        blitEncoder.__fill(tricky.data, range: .init(location: 0, length: tricky.data.length), value: 0)
        blitEncoder.__fill(n1.data, range: .init(location: 0, length: n1.data.length), value: 0)
        blitEncoder.__fill(ssx.data, range: .init(location: 0, length: ssx.data.length), value: 0)
        blitEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    public func outProj(_ ctx: inout OGScratchContext<OG130Spec>, ln: UInt32) throws {
        var ln = Int(ln)
        guard let ld1 = ctx.ld1 else { throw MambaRunnerError.missingData("ld1") }
        guard let ld2 = ctx.ld2 else { throw MambaRunnerError.missingData("ld2") }
        guard let opW = state.layers[ln].out_proj_weight else { throw MambaRunnerError.missingData("outProjW") }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "outProj@\(ln)"
        
        guard let blitEncoder = commandBuffer.makeBlitCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        blitEncoder.__fill(ld1.data, range: .init(location: 0, length: ld1.data.length), value: 0)
        blitEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        
        
        let accursedDepths = tgMaxGroups(totalLengthToDispatch: Int(hp.dInner), max: device.maxThreadsPerThreadgroup.depth)
        var zeds = try vectorBuffer(Array<Float32>.init(repeating: 0.0, count: Int(hp.dModel) * Int(ctx.L)))
        
        
        // when in doubt just make six trillion command buffers i guess
        for ad in accursedDepths {
            guard let commandBuffer2 = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
            commandBuffer2.label = "outProjSuperfluous"
            guard let outProjEncoder = commandBuffer2.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
            outProjEncoder.label = "outProjEncoder"
            outProjEncoder.setComputePipelineState(outProjPS)
            
            outProjEncoder.useHeaps([ctx.heap, state.heap])
            outProjEncoder.useResource(ld1.data, usage: .write)
            outProjEncoder.useResource(ld2.data, usage: .read)
            outProjEncoder.useResource(opW.data, usage: .read)
            
            var L = ctx.L;
            outProjEncoder.setBuffer(ctx.argBuf,        offset: 0,                      index: 0)
            outProjEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
            outProjEncoder.setBuffer(state.stateArgBuffer,        offset: 0,            index: 2)
            outProjEncoder.setBytes(&ln,        length: MemoryLayout<UInt32>.stride,    index: 3)
            outProjEncoder.setBytes(&L,        length: MemoryLayout<UInt32>.stride,     index: 4)
            
            var offset = ad.offset
            let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: ad.dispatchLength)
            outProjEncoder.setBytes(&offset,        length: MemoryLayout<UInt32>.stride,     index: 5)
            
            outProjEncoder.setBuffer(zeds,        offset: 0,                      index: 6)
            
            let threadgroupsPerGrid = MTLSize(width: Int(hp.dModel) / threadsPerThreadgroup.width, height: Int(ctx.L) / threadsPerThreadgroup.height, depth: 1)
            
            outProjEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            outProjEncoder.endEncoding()
            commandBuffer2.commit()
            commandBuffer2.waitUntilCompleted()
        }
        
    }
    
    public func ssm(_ ctx: inout OGScratchContext<OG130Spec>, ln: UInt32) throws {
        var ln = Int(ln)
        guard let A = state.layers[ln].A else { throw MambaRunnerError.missingData("A") }
        guard let commandBuffer  = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "ssm@\(ln)"
        
        /// Compute ∆ A B C D, the state space parameters.
        ///     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        ///     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        ///                                  and is why Mamba is called **selective** state spaces)
        guard let x = ctx.dbld_in else { throw MambaRunnerError.missingData("dbld_in") }
        guard let debici = ctx.debici else { throw MambaRunnerError.missingData("debici") }
        
        let xProjToDBCPS = self.xProjToDBCPS
        var outHeight = ctx.L;
        let outWidth  = Int(hp.dXProj);
        let threadsPerOutZ = Int(hp.dInner);
        
        
        
        let dispatchZGroups = tgMaxGroups(totalLengthToDispatch: threadsPerOutZ,
                                          max: xProjToDBCPS.maxTotalThreadsPerThreadgroup);
        
        let tgpg                     = MTLSize(width: outWidth, height: outHeight, depth: 1)
        
        for var dg in dispatchZGroups {
            guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
            computeEncoder.label = "xProjToDBCPS/\(dg.dispatchLength)@\(dg.offset)"
            computeEncoder.setComputePipelineState(xProjToDBCPS)
            
            computeEncoder.useHeaps([ctx.heap, state.heap])
            computeEncoder.useResource(debici.data, usage: .write)
            computeEncoder.setBuffer(ctx.argBuf,        offset: 0,                      index: 0)
            computeEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
            computeEncoder.useResource(x.data, usage: .read)
            computeEncoder.setBuffer(state.stateArgBuffer,        offset: 0,            index: 2)
            computeEncoder.setBytes(&ln,        length: MemoryLayout<UInt32>.stride,    index: 3)
            computeEncoder.setBytes(&outHeight, length: MemoryLayout<UInt32>.stride,    index: 4)
            computeEncoder.setBytes(&dg.offset, length: MemoryLayout<UInt32>.stride,    index: 5)
            
            let zeds = try vectorBuffer(Array<Float32>.init(repeating: 0.0, count: outWidth * outHeight * (dg.dispatchLength / xProjToDBCPS.threadExecutionWidth)))
            computeEncoder.setBuffer(zeds,        offset: 0,            index: 6)
//            let zeds2 = try vectorBuffer(Array<Float32>.init(repeating: 0.0, count: outWidth * outHeight * (dg.dispatchLength / xProjToDBCPS.threadExecutionWidth) * 2))
//            computeEncoder.setBuffer(zeds,        offset: 0,            index: 7)
//            computeEncoder.setThreadgroupMemoryLength(MemoryLayout<Float32>.stride * xProjToDBCPS.threadExecutionWidth, index: 0)
//
            let threadsPerThreadgroup   = MTLSize(width:         1, height: 1, depth: dg.dispatchLength)
            
            computeEncoder.dispatchThreadgroups(tgpg, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
        }
        
        // Handle dtProj
        guard let deli = ctx.deli else { throw MambaRunnerError.missingData("deli") }
        guard let dtpW = state.layers[ln].dt_proj_weight else { throw MambaRunnerError.missingData("dtProjW") }
        guard let dtpB = state.layers[ln].dt_proj_bias else { throw MambaRunnerError.missingData("dtProjB") }
        guard let dtProjEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        dtProjEncoder.label = "dtProjEncoder"
        dtProjEncoder.setComputePipelineState(dtProjPS)
        
        dtProjEncoder.useHeaps([ctx.heap, state.heap])
        dtProjEncoder.useResource(debici.data, usage: .read)
        dtProjEncoder.useResource(deli.data, usage: .write)
        dtProjEncoder.useResource(dtpW.data, usage: .read)
        dtProjEncoder.useResource(dtpB.data, usage: .read)
        dtProjEncoder.setBuffer(ctx.argBuf,        offset: 0,                      index: 0)
        dtProjEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
        dtProjEncoder.setBuffer(state.stateArgBuffer,        offset: 0,            index: 2)
        dtProjEncoder.setBytes(&ln,        length: MemoryLayout<UInt32>.stride,    index: 3)
        
        // TODO Maximize threadgroup occupancy
        let threadsPerThreadgroup = MTLSize(width: 1, height: 1, depth: Int(hp.dtRank))
        // L, dInner
        let threadgroupsPerGrid = MTLSize(width: Int(hp.dInner), height: Int(ctx.L), depth: 1)
        dtProjEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        dtProjEncoder.endEncoding()
        
        
        // Dispatch one more time to softplus __DELI__
        guard let softplusEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        softplusEncoder.label = "softplusEncoder"
        softplusEncoder.setComputePipelineState(softplusPS)
        softplusEncoder.useHeap(ctx.heap)
        softplusEncoder.useResource(deli.data, usage: .write)
        softplusEncoder.setBuffer(deli.data, offset: 0, index: 0)
        var rowWidth = hp.dInner
        softplusEncoder.setBytes(&rowWidth, length: MemoryLayout<UInt32>.stride, index: 1)
        
        let softTpTg = MTLSize(width: softplusPS.maxTotalThreadsPerThreadgroup, height: 1, depth: 1)
        let softTpg = MTLSize(width: Int(rowWidth), height: Int(ctx.L), depth: 1)
        softplusEncoder.dispatchThreads(softTpg, threadsPerThreadgroup: softTpTg)
        
        softplusEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    public func discretize(_ ctx: inout OGScratchContext<OG130Spec>, ln: UInt32, lCur: Int) throws {
        guard let dA_l  = ctx.dA_l  else { throw MambaRunnerError.missingData("dA_l")  }
        guard let dBu_l = ctx.dBu_l else { throw MambaRunnerError.missingData("dBu_l") }
        guard let deli  = ctx.deli  else { throw MambaRunnerError.missingData("deli") }
        guard let dbld_in = ctx.dbld_in else { throw MambaRunnerError.missingData("dbld_in") }
        guard let debici  = ctx.debici  else { throw MambaRunnerError.missingData("debici") }
        guard let A = state.layers[Int(ln)].A else { throw MambaRunnerError.missingData("layer\(ln).A") }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "discretize@\(ln)"
        guard let discretizeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        discretizeEncoder.setComputePipelineState(discretizePS)
        discretizeEncoder.useHeaps([ctx.heap, state.heap])
        discretizeEncoder.useResource(dA_l.data,  usage: .write)
        discretizeEncoder.useResource(dBu_l.data, usage: .write)
        
        var ln   = ln;
        var lCur = lCur;
        var L    = ctx.L;
        discretizeEncoder.setBuffer(ctx.argBuf,   offset: 0,                         index: 0)
        discretizeEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride, index: 1)
        discretizeEncoder.setBuffer(state.stateArgBuffer, offset: 0,                 index: 2)
        discretizeEncoder.setBytes(&ln,   length: MemoryLayout<UInt32>.stride,       index: 3)
        discretizeEncoder.setBytes(&L,    length: MemoryLayout<UInt32>.stride,       index: 4)
        discretizeEncoder.setBytes(&lCur, length: MemoryLayout<UInt32>.stride,       index: 5)
        
        guard (discretizePS.threadExecutionWidth % Int(hp.dState)) == 0 else { throw MambaRunnerError.invalidParameterShape("dState evenly divides warp size until proven otherwise") }
        let threadsPerThreadgroup = MTLSize(width: discretizePS.threadExecutionWidth / Int(hp.dState), height: Int(hp.dState), depth: 1)
        let threadsPerGrid = MTLSize(width: Int(hp.dInner), height: Int(hp.dState), depth: 1)
        
        discretizeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        discretizeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        print("ooooo Yarg")
        
    }
    
    public func ySiluRes(_ ctx: inout OGScratchContext<OG130Spec>) throws {
        guard let tricky = ctx.tricky else { throw MambaRunnerError.missingData("tricky") }
        guard let dbld_in = ctx.dbld_in else { throw MambaRunnerError.missingData("dbld_in") }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "ySiluRes"
        guard let ySiluRes = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        ySiluRes.setComputePipelineState(ySiluResPS)
        ySiluRes.useHeaps([ctx.heap, state.heap])
        ySiluRes.useResource(tricky.data,      usage: .write)
        ySiluRes.useResource(dbld_in.data,     usage: .read)
        
        var L    = ctx.L;
        
        ySiluRes.setBuffer(ctx.argBuf,   offset: 0,                           index: 0)
        ySiluRes.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
        ySiluRes.setBuffer(state.stateArgBuffer, offset: 0,                   index: 2)
        ySiluRes.setBytes(&L,    length: MemoryLayout<UInt32>.stride,           index: 3)
        
        let tgW = device.maxThreadsPerThreadgroup.width / ySiluResPS.threadExecutionWidth
        let tgH = ySiluResPS.maxTotalThreadsPerThreadgroup / tgW
        let threadsPerThreadgroup = MTLSize(width: tgW,   height: tgH, depth: 1)
        let threadsPerGrid = MTLSize(width: Int(hp.dInner),  height: L, depth: 1)
        
        ySiluRes.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        ySiluRes.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        print("fin yarg this piece of")
        
    }
    public func discretizeAndScanSSM(_ ctx: inout OGScratchContext<OG130Spec>, ln: UInt32, lCur: Int) throws {
        guard let dA_l  = ctx.dA_l  else { throw MambaRunnerError.missingData("dA_l")  }
        guard let dBu_l = ctx.dBu_l else { throw MambaRunnerError.missingData("dBu_l") }
        guard let deli  = ctx.deli  else { throw MambaRunnerError.missingData("deli") }
        guard let tricky = ctx.tricky else { throw MambaRunnerError.missingData("tricky") }
        guard let dbld_in = ctx.dbld_in else { throw MambaRunnerError.missingData("dbld_in") }
        guard let debici  = ctx.debici  else { throw MambaRunnerError.missingData("debici") }
        guard let A = state.layers[Int(ln)].A else { throw MambaRunnerError.missingData("layer\(ln).A") }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "discretizeAndScanSSM@\(ln)"
        guard let dasSSM = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        dasSSM.setComputePipelineState(dasSSMPS)
        dasSSM.useHeaps([ctx.heap, state.heap])
        dasSSM.useResource(dA_l.data,  usage: .read)
        dasSSM.useResource(dBu_l.data, usage: .read)
        dasSSM.useResource(deli.data,  usage: .read)
        dasSSM.useResource(tricky.data,      usage: .write)
        dasSSM.useResource(dbld_in.data,     usage: .read)
        dasSSM.useResource(debici.data,      usage: .read)
        dasSSM.useResource(A.data,           usage: .read)
        
        var ln   = ln;
        var L    = ctx.L;
        var dispatchWidth = 1//self.maxSSMWidth
        
        dasSSM.setBuffer(ctx.argBuf,   offset: 0,                           index: 0)
        dasSSM.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,   index: 1)
        dasSSM.setBuffer(state.stateArgBuffer, offset: 0,                   index: 2)
        dasSSM.setBytes(&ln,   length: MemoryLayout<UInt32>.stride,         index: 3)
        dasSSM.setBytes(&L,    length: MemoryLayout<UInt32>.stride,         index: 4)
        dasSSM.setBytes(&dispatchWidth, length: MemoryLayout<UInt32>.stride,index: 5)
        
        let zeds = try vectorBuffer(Array<Float32>.init(repeating: 0.0, count: 48))
        dasSSM.setBuffer(zeds,   offset: 0,                           index: 6)
        
        let w  = MemoryLayout<Float32>.stride * dispatchWidth
        let n  = MemoryLayout<Float32>.stride * Int(hp.dState)
        let wnDbl = MemoryLayout<Float64>.stride * dispatchWidth * Int(hp.dState)
        dasSSM.setThreadgroupMemoryLength(16,   index: 0) // u
        dasSSM.setThreadgroupMemoryLength(16,   index: 1) // dlt
        dasSSM.setThreadgroupMemoryLength(n,    index: 2) // a
        dasSSM.setThreadgroupMemoryLength(n,    index: 3) // b
        dasSSM.setThreadgroupMemoryLength(n,    index: 4) // c
        dasSSM.setThreadgroupMemoryLength(wnDbl,   index: 8) // x
        
        let threadsPerThreadgroup      = MTLSize(width: dispatchWidth,   height: 1, depth: dasSSMPS.threadExecutionWidth)
        let threadgroupsPerGrid        = MTLSize(width: Int(hp.dInner) / dispatchWidth,  height: 1, depth: 1)
        
        dasSSM.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        dasSSM.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        print("ooooo Yarg")
        
    }
    
    public func rmsNormAndCopy(_ ctx: inout OGScratchContext<OG130Spec>) throws -> MTLBuffer {
        guard let ld1 = ctx.ld1 else { throw MambaRunnerError.missingData("ld1") }
        guard let ld2 = ctx.ld1 else { throw MambaRunnerError.missingData("ld2") }
        guard let norm_f = state.base.norm_f else { throw MambaRunnerError.missingData("norm_f") }
        var rowWidth = Int(hp.dModel)
        var rowCount = ctx.L
        let rowSumPS = self.rowSquaredSumsPS
        guard let commandBuffer = commandQueue.makeCommandBuffer() else { throw MambaRunnerError.failedToMakeCommandBuffer }
        commandBuffer.label = "rowSquaredSums->rmsNormAndCopy"
        guard var computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        computeEncoder.setComputePipelineState(rowSumPS)
        computeEncoder.useHeaps([ctx.heap, state.heap])
        computeEncoder.useResource(ld1.data, usage: .write)
        computeEncoder.setBuffer(ctx.argBuf,        offset: 0,                          index: 0)
        
        computeEncoder.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,       index: 1)
        
        computeEncoder.useResource(norm_f.data, usage: .read)
        computeEncoder.setBuffer(state.stateArgBuffer,        offset: 0,                index: 2)
        
        let (tgWidth, tgsPerRow) = try simdStripes(origSize: rowWidth, warpSize: rowSumPS.threadExecutionWidth, max: rowSumPS.maxTotalThreadsPerThreadgroup)
        
        let threadsPerThreadgroup = MTLSize(width: tgWidth, height: 1, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: tgsPerRow, height: rowCount, depth: 1)
        
        let zeds = try vectorBuffer(Array<Float32>.init(repeating: 0.0, count: rowCount))
        computeEncoder.setBuffer(zeds,        offset: 0,                                index: 3)
        computeEncoder.useResource(zeds, usage: .write)
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        guard var computeEncoderB = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        
        let rmsNormPS = self.rmsNormAndCopyPS
        
        computeEncoderB.useHeaps([ctx.heap, state.heap])
        computeEncoderB.useResource(ld1.data, usage: .write)
        computeEncoderB.useResource(norm_f.data, usage: .read)
        computeEncoderB.useResource(ld2.data, usage: .write)
        computeEncoderB.useResource(zeds, usage: .write)
        
        computeEncoderB.setBuffer(ctx.argBuf,        offset: 0,                          index: 0)
        computeEncoderB.setBytes(&self.hp, length: MemoryLayout<OGHParams>.stride,       index: 1)
        computeEncoderB.setBuffer(state.stateArgBuffer,        offset: 0,                index: 2)
        computeEncoderB.setBuffer(zeds,        offset: 0,                                index: 3)
        
        computeEncoderB.setComputePipelineState(rmsNormPS)
        computeEncoderB.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        
        computeEncoderB.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return zeds
    }
    
//    public func rmsNorm(_ ctx: inout OGScratchContext<OG130Spec>) throws {
//        guard let ld2s = ctx.ld2?.specShape, ld2s == [UInt32(ctx.L), hp.dModel] else {
//            throw MambaRunnerError.shapeMismatch(ctx.ld2?.specShape ?? [], [UInt32(ctx.L), state.spec.hp.dModel])
//        }
//
//        guard let ld1data = ctx.ld1?.data else {throw MambaRunnerError.missingData("ctx.ld1")}
//        guard let ld2data = ctx.ld2?.data else {throw MambaRunnerError.missingData("ctx.ld2")}
//        let cap = ctx.L * Int(hp.dModel)
//        var ld2ptr = ld2data.contents().bindMemory(to: Float16.self, capacity: cap)
//        ld1data.contents().withMemoryRebound(to: Float16.self, capacity: cap) { val in
//            var val = val
//            for _ in 0..<cap {
//                ld2ptr.update(from: val, count: 1)
//                val = val.successor()
//                print(ld2ptr.pointee.formatted())
//                ld2ptr = ld2ptr.successor()
//            }
//        }
//        print("Yarg")
//    }
    
    public func embed(_ ctx: inout OGScratchContext<OG130Spec>) throws {
        let toks = ctx.tokens.filter { 0 <= $0 && $0 < hp.nVocab }
        if (toks.count != ctx.tokens.count) {
            print("Warning: skipping \(ctx.tokens.count - toks.count) out of \(ctx.tokens.count) tokens")
        }
        guard let embedding = state.base.embedding?.data else {
            throw MambaRunnerError.missingData("base.embedding")
        }
        
        let L = ctx.L
        let nVb = try scalarBuffer(UInt32(state.spec.hp.nVocab))
        let dMb = try scalarBuffer(UInt32(state.spec.hp.dModel))
        let sLb = try scalarBuffer(L)
        
        let tokBuf = try vectorBuffer( toks )
        
        let computePipelineState = embedPS
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        commandBuffer.label = "embed"
        
        guard let ld1s = ctx.ld1?.specShape, ld1s == [UInt32(L), state.spec.hp.dModel] else {
            throw MambaRunnerError.shapeMismatch(ctx.ld1?.specShape ?? [], [UInt32(L), state.spec.hp.dModel])
        }
        guard let ld1 = ctx.ld1?.data else { throw MambaRunnerError.missingData("ld1") }
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaRunnerError.failedToMakeCommandEncoder }
        computeEncoder.setComputePipelineState(computePipelineState)
        
        computeEncoder.useHeaps([ctx.heap, state.heap])
        computeEncoder.useResource(ld1, usage: .write)
        
        computeEncoder.setBuffer(tokBuf,    offset: 0, index: 0)
        computeEncoder.setBuffer(ld1,    offset: 0, index: 1)
        computeEncoder.setBuffer(embedding, offset: 0, index: 2)
        computeEncoder.setBuffer(sLb,       offset: 0, index: 3)
        computeEncoder.setBuffer(nVb,       offset: 0, index: 4)
        computeEncoder.setBuffer(dMb,       offset: 0, index: 5)
        
        let w = computePipelineState.threadExecutionWidth
        let h = computePipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        let threadsPerGrid = MTLSize(width: Int(state.spec.hp.dModel), height: Int(L), depth: 1)
        
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
