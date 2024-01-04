import ArgumentParser
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import BPETokenizer

public enum MambaError: Error {
    case invalidFile(String),
         invalidParameterShape(String),
         missingEdgeLayers,
         incompleteLayer,
         unknownLayer,
         stateValidationError,
         failedToMakeCommandBuffer,
         failedToMakeMetalBuffer,
         embeddingNotFound
}

@main
struct Mamba: ParsableCommand {
    @Argument(help: "The folder containing exploded layer weights")
    var folderUrl: String

    @Option(name: .shortAndLong, help: "Prompt text")
    var promptText: String = "Happy New"
    
    @Option(name: .shortAndLong, help: "Number of tokens to generate")
    var tokenCount: Int = 10
    
    
    mutating func run() throws {
        guard var device = MTLCreateSystemDefaultDevice(),
        var q = device.makeCommandQueue(),
        var cmdBuf = q.makeCommandBuffer()
        else {
            print("Your system does not support Metal.")
            throw MambaError.failedToMakeCommandBuffer
        }
        guard let bundleURL = Bundle.module.url(forResource: "MmMamba", withExtension: "metallib") else {
            fatalError("Can't find metal library")
        }
        let library = try device.makeLibrary(URL: bundleURL)
        
        var lmHeadMat: MPSMatrix? = nil
        var embBuf: MTLBuffer? = nil
        
        // Try enumerating contents of folderUrl
        let contents = try FileManager.default.contentsOfDirectory(atPath: folderUrl).filter({$0 != ".DS_Store"})
        
        
        let MSB = MambaStateBuilder()
        for cont in contents
        {
            let metadataPath = folderUrl + "/" + cont + "/metadata.json"
            let metadataJson = try String(contentsOfFile: metadataPath)
            let metadata = try JSONDecoder().decode(MambaLayerMeta.self, from: metadataJson.data(using: .utf8)!)
            
            let binDataPath = folderUrl + "/" + cont + "/weights.bin"
            let buffer = try loadBinaryAsMetalBuffer(binDataPath: binDataPath, device: device, metadata: metadata)
            let tensorData = MPSGraphTensorData(buffer,
                                                shape: metadata.shape,
                                                dataType: .float32,
                                                rowBytes: MemoryLayout<Float32>.size * Int(exactly: metadata.shape.last!)!)

            
            switch(metadata.category)
            {
            case .lm_head:
                lmHeadMat = MPSMatrix(buffer: buffer, descriptor: MPSMatrixDescriptor(rows: 50280,
                                                                                     columns: 768,
                                                                                     rowBytes: 768 * MemoryLayout<Float32>.stride,
                                                                                     dataType: .float32))
                MSB.lmHead = tensorData
                print("Loaded LMHead: shape \(tensorData.shape)")
            case .norm_f:
                MSB.normF = tensorData
                print("Loaded normF: shape \(tensorData.shape)")
            case .embedding:
                embBuf = buffer
                MSB.embeddings = tensorData
                print("Wrote embeddings: shape \(tensorData.shape)")
            case .layers:
                let lb = MSB.getLayerBuilder(index: metadata.index!)
                try lb.addTensorData(tensorData, metadata)
                print("Loaded \(cont): shape \(tensorData.shape) -> \(previewFloatTensorBytes(tensorData, offset: 0, n: 2))...")
            }
        }
        
        
        
        var state = try MSB.validated()
        state.embBuf = embBuf!
        let graph = MambaMPSGraph()
        let model = MambaRunner(state: state, device: device, cmdQ: q, library: library)
                
        print("Prompt: \(promptText)")
        
        // "My name is"
        let specialTokensMapPath = Bundle.module.url(forResource:"special_tokens_map", withExtension:"json")
        let tokenizerPath = Bundle.module.url(forResource:"tokenizer", withExtension:"json")
        let tokenizer = try BPETokenizer(pathToTokenizerConfig: tokenizerPath!, pathToSpecialTokensMap: specialTokensMapPath!)
        let initialTokens: [UInt32] = tokenizer.tokenize(promptText).map { UInt32($0.tokenId) }
        
        try model.embed(initialTokens, embeddings: embBuf!)
        
////        let tokenEmbeddings = model.embed(initialTokens, embeddings: &embBuf!)
//        
//        cmdBuf.commit()
        
//        *  @abstract   Applies a gather operation along a given axis.  The encoded primary source array
//        *              contains the data and the secondary array is a 1-D MPSNDArray containing the
//        *              indices.
//        *
//        *                  For each dimension other than axis
//        *                      result[i] = source[i]; 0 <= i < array slice length along dimension
//        *                  Along the specified axis
//        *                      result[i] = source[indices[i]]; 0 <= i < number of indices

//        for _ in 0..<tokenCount {
//            tokens = try generate(graph, device:device, state: state, inTokens: tokens)
//            print(tokenizer.unTokenize(tokens))
//        }
        print("Done")
    }
    
    
    
    func generate(_ graph: MambaMPSGraph, device: MTLDevice, state: MambaState, inTokens: [Int32]) throws -> [Int32] {
        var broadTokens = inTokens.map {
            Array<Int32>.init(repeating: $0, count: Int(truncating: state.embeddings.shape[1]))
        }.flatMap {$0}
        
        var tokenArray = MPSNDArray(device: device,
                                    descriptor: MPSNDArrayDescriptor(
                                        dataType: .int32,
                                        shape: [NSNumber(value: broadTokens.count / Int(truncating: state.embeddings.shape[1])),
                                                state.embeddings.shape[1] ] ) )
        
        broadTokens.withUnsafeMutableBytes { ptr in
            tokenArray.writeBytes(ptr.baseAddress!, strideBytes: nil)
        }

        let tokenData = MPSGraphTensorData(tokenArray)
        
        let inTokensPlaceholder   = graph.placeholder(shape: tokenData.shape, dataType: .int32,   name: "inputTokensP")
        let embeddingsPlaceholder = graph.placeholder(shape: state.embeddings.shape, dataType: .float32, name: "embeddingsP")
        let normFPlaceholder  = graph.placeholder(shape: state.normF.shape, dataType: .float32, name: "normFP")
        
        let operands = state.layers.map { layerState in MambaBlockOperand(state: layerState, graph: graph) }
        let emb      = graph.embeddingsOf(inTokensPlaceholder, within: embeddingsPlaceholder)
        let margs    = MambaArgs()
        let laysors  = try operands.reduce(into: emb) { x, layer in
            x = try graph.residual(passing: x, into: layer, margs: margs)
        }
        let normFoo = graph.rmsNorm(laysors, weights: normFPlaceholder)
        let logits  = graph.matrixMultiplication(primary: normFoo,
                                                secondary: graph.transposeTensor(embeddingsPlaceholder, dimension: 0, withDimension:1, name: "logits.transpose"),
                                                name: "logits.matmul")
        let probs   = graph.softMax(with:
                                    graph.sliceTensor(logits,
                                                      dimension: 0,
                                                      start: Int(truncating: logits.shape![0]) - 1,
                                                      length: 1,
                                                      name:nil),
                                  axis: -1,
                                  name: nil)
        let topK = graph.topK(probs, k: 40, name: nil)
        let aboveBar = graph.lessThan(probs, graph.reductionMinimum(with: topK[0], axis: 1, name: nil) , name: nil)
        // MPS select only supports tensor dimensions in range N[1-65536]D[1-16384]C[1-65536]H[1-16384]W[1-16384]
        let cutP = graph.select(predicate: graph.reshape(aboveBar, shape: [NSNumber(value: margs.vocabSize), 1, 1], name: nil),
                                trueTensor: graph.reshape(probs,shape: [NSNumber(value: margs.vocabSize), 1, 1], name: nil),
                                falseTensor: graph.constant(0.0, shape: [NSNumber(value: margs.vocabSize), 1, 1], dataType: .float32),
                                name: nil)
        let cutPr = graph.reshape(cutP, shape: [NSNumber(value: margs.vocabSize)], name:nil)
        
        print("Running MPS Graph...")
        let results = graph.run(
            feeds: [inTokensPlaceholder:tokenData,
                  embeddingsPlaceholder:state.embeddings,
                       normFPlaceholder:state.normF,
                   ].merging(operands.flatMap({$0.feeds}), uniquingKeysWith: {$1}),
            targetTensors: [cutPr],
            targetOperations: nil
        )
        
        peekFloatTensorFloats(results[cutPr])
        let arr = results[cutPr]!.mpsndarray()
        let totalSize = (0..<arr.numberOfDimensions)
            .reduce(into: 1, {$0 *= arr.length(ofDimension: $1)})
        
        
        var values: [Float32] = .init(repeating: -42.0, count: totalSize)
        arr.readBytes(&values, strideBytes: nil)
        values = values.map{ $0.isNaN ? 0 : $0 }
        
        
        // Sum of all probabilities (so that we don't have to require that the sum is 1.0):
        let sum = values.reduce(0, +)
        // Random number in the range 0.0 <= rnd < sum :
        let rnd = Float32.random(in: 0.0 ..< sum)
        var choose = values.count - 1
        
        // Find the first interval of accumulated probabilities into which `rnd` falls:
        var accum: Float32 = 0.0
        for (i, p) in values.enumerated() {
            accum += p
            if rnd < accum {
                choose = i
                break
            }
        }
        print(choose)
        return inTokens + [Int32(choose)]
    }
    
}


public class MambaMPSGraph: MPSGraph {
    func selectiveScan(_ u: MPSGraphTensor,
                       delta: MPSGraphTensor,
                       A: MPSGraphTensor,
                       B: MPSGraphTensor,
                       C: MPSGraphTensor,
                       D: MPSGraphTensor) -> MPSGraphTensor {
        let l = u.shape![0]
        let dIn = u.shape![1]
        let n = A.shape![1]
        
        
        let deltaRows = split(delta, splitSizes: .init(repeating: 1, count: Int(truncating: dIn)), axis: 1, name: "rows")
        let ACols = split(A    , splitSizes: .init(repeating: 1, count: Int(truncating: dIn)), axis: 0, name: "rows")
        let deltaACube: MPSGraphTensor = exponent(with: stack(
            deltaRows.enumerated().map { (i, r) in
                matrixMultiplication(primary: r, secondary: ACols[i], name: nil)
            },
            axis: 0,
            name: "cube"),
                                                  name: "exp")
        //        let deltaB_uCube =
        let u2 = reshape(u, shape: [l, dIn], name: nil)
        let dLix = split(delta, splitSizes: .init(repeating: 1, count: Int(truncating: l)), axis: 0, name: "Delta's lix")
        let bLix = split(B, splitSizes: .init(repeating: 1, count: Int(truncating: l)), axis: 0, name: "B's lix")
        let uLix = split(u2, splitSizes: .init(repeating: 1, count: Int(truncating: l)), axis: 0, name: "U's lix")
        let deltaB_uCube = stack(
            dLix.enumerated().map { (lix, d™) in
                let dux = transpose(multiplication(d™, uLix[lix], name: nil), permutation: [1, 0], name: nil) // (dIn, 1)
                return matrixMultiplication(primary: dux, secondary: bLix[lix], name: nil) // (dIn, n)
            },
            axis: 1,
            name: "stackBuCube"
        ) // (dIn, l, n)
        
        
        let xony = constant(0.0, shape: [dIn, n], dataType: .float32)
        let daLix = split(
            transpose(deltaACube, permutation: [1, 0, 2], name: nil), // (dIn, l, n) -> (l, dIn, n)
            splitSizes: .init(repeating: 1, count: Int(truncating: l)),
            axis: 0,
            name: "DeltaACube's lix")
        let dbuLix = split(deltaB_uCube, splitSizes: .init(repeating: 1, count: Int(truncating: l)), axis: 1, name: "DeltaBuCube's lix")
        let cLix = split(C, splitSizes: .init(repeating: 1, count: Int(truncating: l)), axis: 0, name: "C's lix")
        
        let (_, xany) = daLix.reduce(into: (0, xony), ({ (acc, da™) in
            let (i, vals) = acc
            let xyoink = split(vals, splitSizes: [n, NSNumber(value: i)], axis: 1, name: "xiPrev")
            let xiPrev = xyoink[0]
            let yiPrev = xyoink[1]
            let xi = addition(
                multiplication(squeeze(da™, name:nil), xiPrev, name:nil),
                squeeze(dbuLix[i], name:nil),
                name:nil) // (dIn, n)
            let yi4me = matrixMultiplication(primary: xi, secondary: transpose(cLix[i], permutation: [1, 0], name:nil), name:nil) // (dIn,)
            let yi = concatTensor(yiPrev,
                                  with: reshape(yi4me, shape: [dIn, 1], name: nil),
                                  dimension: 1,
                                  name: nil) // (dIn, i)
            let xouchy = concatTensor(xi, with: yi, dimension: 1, name: nil) // (dIn, n + i)
            acc = (i + 1, xouchy)
        }))
        
        let y = split(xany, splitSizes: [n, l], axis: 1, name: nil)[1]
        
        
        return addition(transpose(y, permutation: [1,0], name:nil),
                        multiplication(squeeze(u, name:nil), D, name: nil),
                        name:nil
        )
    }
    
    func softplus(_ x: MPSGraphTensor) -> MPSGraphTensor {
        return logarithm(
            with: addition(constant(1, shape: [1], dataType: .float32), exponent(with: x, name: "exp(x)"),
                           name: "1 + exp(x)"),
            name: "log(1 + exp(x)"
            )
    }
    
    func silu(_ x: MPSGraphTensor) -> MPSGraphTensor {
        return multiplication(x, sigmoid(with: x, name: "silu"), name: "silu")
    }
    
    func mamba(_ x: MPSGraphTensor, into layer: MambaBlockOperand, margs: MambaArgs) throws -> MPSGraphTensor {
        // .shape: (l, d)
        guard let l = x.shape?[0], let _ = x.shape?[1] else {
            throw MambaError.invalidParameterShape("mamba/x@\(layer.index)")
        }
        
        /// Projection of shape (l, 2(dInner)), where left is x and right is residual
        let xAndRes = matrixMultiplication(primary: x,
                                           secondary: transposeTensor(layer.inProj.key,
                                                                      dimension: 0,
                                                                      withDimension: 1,
                                                                      name: "mamba:xAndRes.transpose@\(layer.index)"),
                                           name: "mamba:xAndRes.matmul@\(layer.index)")
        let xrSplit = split(xAndRes, splitSizes: [margs.dInner, margs.dInner], axis: -1, name: "mamba:xrSplit.split@\(layer.index)")
        let x   = xrSplit[0] // (l, dInner)
        let res = xrSplit[1] // (l, dInner)
        
        let dConv: NSNumber = 4
        let dtRank: NSNumber = 48
        
        // x (l,dInner) -> (dInner, l)
        let trans = transpose(x, permutation: [1,0], name: "mamba:trans.transpose@\(layer.index)")
        
        // Conv Input: (N:1, C:dInner, H:1, W:L)
        let xPanded = reshape(trans, shape: [1, margs.dInner, 1, l], name: "mamba:xPanded.reshape@\(layer.index)")
        
        // Conv Kernel: (N:1, C:dInner, H:1, W:dConv)
        let weXpand = reshape(layer.conv1dWeight.key, shape: [1, margs.dInner, 1, dConv], name: "mamba:weXpand.reshape@\(layer.index)")
        
        // Padding explicitly (dConv - 1) to match PyTorch output shape
        let descriptor = MPSGraphDepthwiseConvolution2DOpDescriptor(strideInX: 1,
                                                                    strideInY: 1,
                                                                    dilationRateInX: 1,
                                                                    dilationRateInY: 1,
                                                                    paddingLeft: Int(truncating: dConv) - 1,
                                                                    paddingRight: Int(truncating: dConv) - 1,
                                                                    paddingTop: 0,
                                                                    paddingBottom: 0,
                                                                    paddingStyle: .explicit,
                                                                    dataLayout: .NCHW,
                                                                    weightsLayout: .NCHW)
        // New shape after convolution: (N:1, C:dInner, H:1, W:((dConv - 1) + L))
        let conv2d = depthwiseConvolution2D(xPanded,
                                            weights: weXpand,
                                            descriptor: descriptor!,
                                            name: "mamba:conv2d.depthwiseConvolution2D@\(layer.index)")
        // First of what is sure to be many Ls
        let firstL = sliceTensor(conv2d, dimension: 3, start: 0, length: Int(truncating: l), name: "mamba:firstL.sliceTensor@\(layer.index)")
        let biased = addition(firstL,
                              reshape(layer.conv1dBias.key, shape: [1, margs.dInner, 1, 1], name: "mamba:biased.reshape@\(layer.index)"),
                              name: "mamba:biased.addition@\(layer.index)")
        let transpose2 = transposeTensor(biased, dimension: 0, withDimension: 3, name: "mamba:transpose2.transposeTensor@\(layer.index)")
        let silt = silu(transpose2)
        
        let yesm = try ssm(passing: silt, into: layer, margs: margs)
        
        let yup  = multiplication(yesm, silu(res), name: "mamba:yup.multiplication@\(layer.index)")

        /// Projection of shape (l, 2(dInner)), where left is x and right is residual
        let output = matrixMultiplication(primary: yup,
                                          secondary: transposeTensor(layer.outProj.key,
                                                                      dimension: 0,
                                                                      withDimension: 1,
                                                                      name: "mamba:output.transpose@\(layer.index)"),
                                           name: "mamba:output.matmul@\(layer.index)")

        
        return output
    }
    
    func ssm(passing x: MPSGraphTensor, into layer: MambaBlockOperand, margs: MambaArgs) throws -> MPSGraphTensor {
        let aLogP = layer.aLog.key
        guard aLogP.shape?[0] == margs.dInner, let n = aLogP.shape?[1] else {
            throw MambaError.invalidParameterShape("ssm/aLogP@\(layer.index)")
        }
        guard let l = x.shape?[0] else {
            throw MambaError.invalidParameterShape("ssm/x@\(layer.index)")
        }
        
        /// Compute ∆ A B C D, the state space parameters.
        ///     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        ///     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        ///                                  and is why Mamba is called **selective** state spaces)
        let negExp = exponent(with: aLogP, name: "ssm:negExp.exponent@\(layer.index)")
        let A = negative(with: negExp, name: "ssm:A.negative@\(layer.index)")
        let D = layer.D.key
        
        let xDbl = matrixMultiplication(primary: reshape(x, shape:[l,margs.dInner], name:"ssm:xDbl.reshape@\(layer.index)"),
                                        secondary: transpose(layer.xProj.key, permutation: [1,0], name: "ssm:xDbl.transpose@\(layer.index)"),
                                        name: "ssm:xDbl.matmul@\(layer.index)")
        
        // (L, dtRank * 2*n)
        let dbc = split(xDbl, splitSizes: [margs.dtRank, n, n], axis: -1, name: "ssm:dbc.split@\(layer.index)")
        
        guard let delta = try? dbc[0], let B = try? dbc[1], let C = try? dbc[2] else {
            throw MambaError.invalidParameterShape("ssm/delta@\(layer.index)")
        }
        
        let dtp = matrixMultiplication(primary: delta,
                                             secondary: transpose(layer.dtProjWeight.key, permutation: [1, 0], name: "ssm:dtp.transpose@\(layer.index)"),
                                             name: "ssm:dtp.matmul@\(layer.index)")
        
        let dtpwb = addition(dtp, layer.dtProjBias.key, name: "ssm:dtpwb.addition@\(layer.index)")
        let softplus = softplus(dtpwb)
        
        return selectiveScan(x, delta: softplus, A: A, B: B, C: C, D: D)
    }
    
    func residual(passing x: MPSGraphTensor, into layer: MambaBlockOperand, margs: MambaArgs) throws -> MPSGraphTensor {
        let norm  = rmsNorm(x, weights: layer.norm.key)
        let mambo = try mamba(norm, into: layer, margs: margs)
        return addition(mambo, x, name: "residual:+@\(layer.index)")
    }
    
    func rmsNorm(_ x: MPSGraphTensor, weights: MPSGraphTensor, eps: Double = 1e-5) -> MPSGraphTensor {
        let meanSquare = mean(of: square(with: x, name: "rmsNorm:x^2"), axes: [-1], name: "rmsNorm:mean(x^2)")
        let epses = constant(eps, shape: [1], dataType: .float32)
        let adjMeanSqu = addition(meanSquare, epses, name:"rmsNorm:mean(x^2)+eps")
        let rsqrt = reverseSquareRoot(with: adjMeanSqu, name: "rmsNorm:rsqrt(mean(x^2) + eps)")
        let scaledX = multiplication(x, rsqrt, name: "rmsNorm:x * rsqrt(mean(x^2) + eps)")
        let output = multiplication(scaledX, weights, name: "rmsNorm:output")
        return output
    }
    
    public func embeddingsOf(_ tokens: MPSGraphTensor, within embeddings: MPSGraphTensor) -> MPSGraphTensor {
        return gatherAlongAxis(0, updates: embeddings, indices: tokens, name: "embGather")
    }
    
    
}

public class MambaRunner {
    public let state: MambaState
    public let dModel: Int = 768
    public let nVocab: Int = 50280
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    
    public init(state: MambaState, device: MTLDevice, cmdQ: MTLCommandQueue, library: MTLLibrary) {
        self.state = state
        self.commandQueue = cmdQ
        self.device = device
        self.library = library
    }
    
    func printSmokeTest() throws {
        let kernelFunction = library.makeFunction(name: "smokeTest")!
        let computePipelineState = try device.makeComputePipelineState(function: kernelFunction)
        
        
        let dataSize = 256 // number of elements
        guard let buffer = device.makeBuffer(length: dataSize * MemoryLayout<Float>.size, options: []) else {
            fatalError("Failed to make buffer")
        }
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(buffer, offset: 0, index: 0)
        let threadsPerGroup = MTLSize(width: 16, height: 1, depth: 1)
        let numThreadgroups = MTLSize(width: (dataSize + 15) / 16, height: 1, depth: 1)
        computeEncoder.dispatchThreadgroups(numThreadgroups, threadsPerThreadgroup: threadsPerGroup)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        var output = [Float]()
        let dataPointer = buffer.contents().bindMemory(to: Float.self, capacity: dataSize)
        let bufferPointer = UnsafeBufferPointer(start: dataPointer, count: dataSize)
        bufferPointer.forEach { value in output.append(value) }
        print(output)
        print("Yarg")
    }
    
    public func embed(_ tokenIds: [UInt32], embeddings: MTLBuffer) throws {
        var nVocab: UInt32 = UInt32(nVocab)
        let nVb = device.makeBuffer(bytes: &nVocab, length: MemoryLayout<UInt32>.stride)
        var dModel: UInt32 = UInt32(dModel)
        let dMb = device.makeBuffer(bytes: &dModel, length: MemoryLayout<UInt32>.stride)
        
        
        var toks: [UInt32] = tokenIds
        var seqLen: UInt32 = UInt32(toks.count)
        let sLb = device.makeBuffer(bytes: &seqLen, length: MemoryLayout<UInt32>.stride)
        
        let toksTotalBytes = MemoryLayout<UInt32>.stride * toks.count
        var tokBuf: MTLBuffer? = device.makeBuffer(bytes: toks, length: toksTotalBytes)
        toks.withUnsafeMutableBytes { tb in
            print(tb.baseAddress.debugDescription)
            tokBuf?.contents().copyMemory(from: tb.baseAddress!, byteCount: toksTotalBytes)
        }
        guard let tokBuf = tokBuf else {
            fatalError("Failed to make token buffer")
        }
        
        let embFn = library.makeFunction(name: "getEmbeddings")!
        let computePipelineState = try device.makeComputePipelineState(function: embFn)
    //    customScope.begin()
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let outBufSize = toks.count * Int(dModel)
        let outByteCount = outBufSize * MemoryLayout<Float32>.stride
        var outBuf = device.makeBuffer(length: outByteCount)
        var output = [Float]()
        
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.setComputePipelineState(computePipelineState)
        
        computeEncoder.setBuffer(tokBuf,  offset: 0, index: 0)
        computeEncoder.setBuffer(outBuf,  offset: 0, index: 1)
        computeEncoder.setBuffer(embeddings,  offset: 0, index: 2)
        computeEncoder.setBuffer(sLb,  offset: 0, index: 3)
        computeEncoder.setBuffer(nVb,  offset: 0, index: 4)
        computeEncoder.setBuffer(dMb,  offset: 0, index: 5)
        
        //           0  1  2  3  4  5  6  7  8  9  10 11 12 13 14
        // Example [ ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ?  ? ]
        let w = computePipelineState.threadExecutionWidth
        let h = computePipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        let threadsPerGrid = MTLSize(width: Int(dModel), height: Int(seqLen), depth: 1)
        
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        commandBuffer.commit()
    //    customScope.end()
    //    let captureManager = MTLCaptureManager.shared()
    //    captureManager.stopCapture()
        commandBuffer.waitUntilCompleted()
        
        let dataPointer = outBuf!.contents().bindMemory(to: Float.self, capacity: outBufSize)
        let bufferPointer = UnsafeBufferPointer(start: dataPointer, count: outBufSize)
        bufferPointer.forEach { value in output.append(value) }
        print(output)
        print("Yarg")
    }
    
    
//    func rmsNorm(_ x: MPSNDArray, weights: MPSMatrix, eps: Double = 1e-5) -> MPSNDArray {
//        
//        //        let meanSquare = mean(of: square(with: x, name: "rmsNorm:x^2"), axes: [-1], name: "rmsNorm:mean(x^2)")
////        let epses = constant(eps, shape: [1], dataType: .float32)
////        let adjMeanSqu = addition(meanSquare, epses, name:"rmsNorm:mean(x^2)+eps")
////        let rsqrt = reverseSquareRoot(with: adjMeanSqu, name: "rmsNorm:rsqrt(mean(x^2) + eps)")
////        let scaledX = multiplication(x, rsqrt, name: "rmsNorm:x * rsqrt(mean(x^2) + eps)")
////        let output = multiplication(scaledX, weights, name: "rmsNorm:output")
//        return output
//    }
}
