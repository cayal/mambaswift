import ArgumentParser
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import BPETokenizer
import MamBufLo

public enum MambaError: Error {
    case invalidFile(String),
         unsupportedModel(String),
         invalidParameter(String),
         invalidParameterShape(String),
         missingData(String),
         missingEdgeLayers,
         incompleteLayer,
         unknownLayer,
         stateValidationError,
         failedToMakeCommandBuffer,
         failedToMakeCommandEncoder,
         failedToMakeMetalBuffer,
         failedToMakeMetalHeap,
         embeddingNotFound
}

@main
struct Mamba: ParsableCommand {
    @Argument(help: "The folder containing exploded layer weights")
    var folderUrl: String?

    @Option(name: .shortAndLong, help: "Prompt text")
    var promptText: String = "Happy New"
    
    @Option(name: .shortAndLong, help: "Number of tokens to generate")
    var tokenCount: Int = 10
    
    
    mutating func run() throws {
        let SupportedModels: [any ModelStateSpec] = [OG130()]
        
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
        
        let furl = try folderUrl ?? {
            print("No folder url specified, looking for converted models...")
            guard let convertedModelsPath = Bundle.module.url(forResource: "converted", withExtension: nil)?.path,
                  let availableModels = try? FileManager.default.contentsOfDirectory(atPath: convertedModelsPath).filter({$0 != ".DS_Store"}),
                  availableModels.count > 0 else {
                print("Couldn't find a model to use. Specify a folder URL or convert a model with `convert.py`.")
                throw MambaError.invalidFile("")
            }
            guard let foundName = availableModels.first(where: { avail in SupportedModels.first(where: { $0.name == avail }) != nil }) else {
                print("Supported models are: \(SupportedModels.map{$0.name}.joined(separator: ", ")). Found: \(availableModels.joined(separator: ","))")
                throw MambaError.invalidFile("")
            }
            return convertedModelsPath + "/" + foundName
        }()
        
        // Double check this one more time lol
        let chosenName = furl.split(separator: "/").last ?? ""
        guard let modelToUse = SupportedModels.first(where: { $0.name ==  chosenName }) else {
            print("Passed: \(chosenName). Supported models are: \(SupportedModels.map{$0.name}.joined(separator: ", ")).")
            throw MambaError.invalidFile(furl)
        }
        
        let contents = try FileManager.default.contentsOfDirectory(atPath: furl).filter({$0 != ".DS_Store"})
        print("Oh no")
        
        
        let MSB = MambaStateBuilder()
        let MBL = try MamBufLoBuilder(modelToUse)
        
        for cont in contents
        {
            let metadataPath = furl + "/" + cont + "/metadata.json"
            let metadataJson = try String(contentsOfFile: metadataPath)
//            let metadata = try JSONDecoder().decode(MambaLayerMeta.self, from: metadataJson.data(using: .utf8)!)
            let metadata = try JSONDecoder().decode(InputMetadata.self, from: metadataJson.data(using: .utf8)!)
            let binDataPath = furl + "/" + cont + "/weights.bin"
            //MSB.addState(metadata, binDataPath)
            try MBL.include(metadata, pathOnDisk: binDataPath)
            
//            let buffer = try loadBinaryAsMetalBuffer(binDataPath: binDataPath, device: device, metadata: metadata)
//            let tensorData = MPSGraphTensorData(buffer,
//                                                shape: metadata.shape,
//                                                dataType: .float32,
//                                                rowBytes: MemoryLayout<Float32>.size * Int(exactly: metadata.shape.last!)!)
//
//            
//            switch(metadata.category)
//            {
//            case .lm_head:
//                lmHeadMat = MPSMatrix(buffer: buffer, descriptor: MPSMatrixDescriptor(rows: 50280,
//                                                                                     columns: 768,
//                                                                                     rowBytes: 768 * MemoryLayout<Float32>.stride,
//                                                                                     dataType: .float32))
//                MSB.lmHead = tensorData
//                print("Loaded LMHead: shape \(tensorData.shape)")
//            case .norm_f:
//                MSB.normF = tensorData
//                print("Loaded normF: shape \(tensorData.shape)")
//            case .embedding:
//                embBuf = buffer
//                MSB.embeddings = tensorData
//                print("Wrote embeddings: shape \(tensorData.shape)")
//            case .layers:
//                let lb = MSB.getLayerBuilder(index: metadata.index!)
//                try lb.addTensorData(tensorData, metadata)
//                print("Loaded \(cont): shape \(tensorData.shape) -> \(previewFloatTensorBytes(tensorData, offset: 0, n: 2))...")
//            }
        }
        
        
        
        var state = try MBL.complete(device: device, cmdQ: q)
        let model = MambaRunner(state: state, device: device, cmdQ: q, library: library)
        
//        var state = try MB.validated()
//        state.embBuf = embBuf!
//        let graph = MambaMPSGraph()
                
        print("Prompt: \(promptText)")
        
//         "My name is"
        let specialTokensMapPath = Bundle.module.url(forResource:"special_tokens_map", withExtension:"json")
        let tokenizerPath = Bundle.module.url(forResource:"tokenizer", withExtension:"json")
        let tokenizer = try BPETokenizer(pathToTokenizerConfig: tokenizerPath!, pathToSpecialTokensMap: specialTokensMapPath!)
        var tokens: [Int32] = tokenizer.tokenize(promptText).map { Int32($0.tokenId) }
        var embeddings = try model.embed(tokens)
        
        cmdBuf.commit()
        var embOut: [Float16] = []
        let dataPointer = embeddings.contents().bindMemory(to: Float16.self, capacity: embeddings.length)
        let bufferPointer = UnsafeBufferPointer(start: dataPointer, count: embeddings.length)
        bufferPointer.forEach { value in embOut.append(value) }
        print(embOut)

//        for _ in 0..<tokenCount {
////            tokens = try generate(graph, device:device, state: state, inTokens: tokens)
//            let embeddings = try model.embed(tokens, embeddings: state.embBuf!)
//            let embMatrix  = MPSMatrix(buffer: embeddings, descriptor: MPSMatrixDescriptor(rows: tokens.count,
//                                                                                           columns: model.dModel,
//                                                                                           rowBytes: model.dModel * MemoryLayout<Float32>.stride,
//                                                                                           dataType: .float32))
//            let embTnData = MPSGraphTensorData(embMatrix)
//            let embPlaceh = graph.placeholder(shape: embTnData.shape, dataType: .float32, name: "embPlaceh")
//            let operands = state.layers.map { layerState in MambaBlockOperand(state: layerState, graph: graph) }
//            var xp: MPSGraphTensor = embPlaceh
//            var x: MPSGraphTensorData = embTnData
//            let margs    = MambaArgs()
//            for layer in operands {
//                let norm  = graph.rmsNorm(xp, weights: layer.norm.key)
//                // .shape: (l, d)
//                guard let l = norm.shape?[0], let _ = norm.shape?[1] else {
//                    throw MambaError.invalidParameterShape("mamba/norm@\(layer.index)")
//                }
//                
//                /// Projection of shape (l, 2(dInner)), where left is x and right is residual
//                let normAndRes = graph.matrixMultiplication(primary: norm,
//                                                            secondary: graph.transposeTensor(layer.inProj.key,
//                                                                                             dimension: 0,
//                                                                                             withDimension: 1,
//                                                                                             name: "mamba:normAndRes.transpose@\(layer.index)"),
//                                                            name: "mamba:normAndRes.matmul@\(layer.index)")
//                
//                let normrSplit = graph.split(normAndRes, splitSizes: [margs.dInner, margs.dInner], axis: -1, name: "mamba:normrSplit.split@\(layer.index)")
//                let x2   = normrSplit[0] // (l, dInner)
//                let res = normrSplit[1] // (l, dInner)
//                
//                let results = try genToselScan(graph,
//                                               model: model,
//                                               device: device,
//                                               state: state,
//                                               inTokens: tokens,
//                                               layer: layer,
//                                               x2: x2,
//                                               res: res,
//                                               l: l,
//                                               xp: xp,
//                                               x: x,
//                                               margs: margs
//                )
//                
//                let nextCmdBuf = q.makeCommandBuffer()!
//                let und = results[0].mpsndarray()
//                let spdnd = results[1].mpsndarray()
//                let and = results[2].mpsndarray()
//                let bnd = results[3].mpsndarray()
//                let cnd = results[4].mpsndarray()
//                let dnd = results[5].mpsndarray()
//                var u = device.makeBuffer(length: und.resourceSize())!
//                var spd = device.makeBuffer(length: spdnd.resourceSize())!
//                var a = device.makeBuffer(length: and.resourceSize())!
//                var b = device.makeBuffer(length: bnd.resourceSize())!
//                var c = device.makeBuffer(length: cnd.resourceSize())!
//                var d = device.makeBuffer(length: dnd.resourceSize())!
//                und.exportData(with: nextCmdBuf, to: u, destinationDataType: .float32, offset: 0, rowStrides: nil)
//                spdnd.exportData(with: nextCmdBuf, to: spd, destinationDataType: .float32, offset: 0, rowStrides: nil)
//                and.exportData(with: nextCmdBuf, to: a, destinationDataType: .float32, offset: 0, rowStrides: nil)
//                bnd.exportData(with: nextCmdBuf, to: b, destinationDataType: .float32, offset: 0, rowStrides: nil)
//                cnd.exportData(with: nextCmdBuf, to: c, destinationDataType: .float32, offset: 0, rowStrides: nil)
//                dnd.exportData(with: nextCmdBuf, to: d, destinationDataType: .float32, offset: 0, rowStrides: nil)
//                nextCmdBuf.commit()
//                nextCmdBuf.waitUntilCompleted()
//                
//                let selScanFn = library.makeFunction(name: "selectiveScan")!
//                let computePipelineState = try device.makeComputePipelineState(function: selScanFn)
//                var dInner = UInt32(1536)
//                var dState = UInt32(16)
//                let seqLen = UInt32(tokens.count)
//                let outBufSize = dInner * seqLen
//                let outByteCount = Int(outBufSize) * MemoryLayout<Float32>.stride
//                let outBuf = device.makeBuffer(length: outByteCount)
//                var output = [Float]()
//                
//                let xBufSize = dInner * (dState / 4)
//                let xByteCount = Int(xBufSize) * MemoryLayout<SIMD4<Float>>.stride
//                var xBuf = device.makeBuffer(length: xByteCount)
//                
//                let dInnerB = device.makeBuffer(bytes: &dInner, length: MemoryLayout<UInt32>.stride)
//                let dStateB = device.makeBuffer(bytes: &dState, length: MemoryLayout<UInt32>.stride)
//                
//                //            withDebugCapture(on: device, execute:  {
//                var computeCmdBuf = q.makeCommandBuffer()!
//                var computeEncoder = computeCmdBuf.makeComputeCommandEncoder(dispatchType: .serial)!
//                computeEncoder.setComputePipelineState(computePipelineState)
//                
//                var loopIndices: [UInt32] = [0, 1]
//                let liB = device.makeBuffer(bytes: &loopIndices, length: loopIndices.count * MemoryLayout<UInt32>.stride)
//                computeEncoder.setBuffer(u,  offset: 0, index: 0)
//                computeEncoder.setBuffer(spd,  offset: 0, index: 1)
//                computeEncoder.setBuffer(a,  offset: 0, index: 2)
//                computeEncoder.setBuffer(b,  offset: 0, index: 3)
//                computeEncoder.setBuffer(c,  offset: 0, index: 4)
//                computeEncoder.setBuffer(d,  offset: 0, index: 5)
//                computeEncoder.setBuffer(liB,  offset: 0, index: 6)
//                computeEncoder.setBuffer(dInnerB,  offset: 0, index: 7)
//                computeEncoder.setBuffer(dStateB,  offset: 0, index: 8)
//                computeEncoder.setBuffer(xBuf,    offset: 0, index: 9)
//                computeEncoder.setBuffer(outBuf,  offset: 0, index: 10)
//                
//                for li in loopIndices.map({Int($0)}) {
//                    computeEncoder.setBufferOffset(4 * li * Int(dInner), index: 0)
//                    computeEncoder.setBufferOffset(4 * li * Int(dInner), index: 1)
//                    computeEncoder.setBufferOffset(4 * li * Int(dState), index: 3)
//                    computeEncoder.setBufferOffset(4 * li * Int(dState), index: 4)
//                    computeEncoder.setBufferOffset(4 * li, index: 6)
//                    let h = computePipelineState.threadExecutionWidth
//                    let w = 1
//                    let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
//                    let threadsPerGrid = MTLSize(width: Int(dInner), height: Int(dState), depth: 1)
//                    computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
//                }
//                
//                computeEncoder.endEncoding()
//                computeCmdBuf.commit()
//                computeCmdBuf.waitUntilCompleted()
//                //            })
//                
//                let yPlaceholder = graph.placeholder(shape: [NSNumber(value: seqLen), NSNumber(value: dInner)], dataType: .float32, name: "yPlaceh")
//                let yTensorData = MPSGraphTensorData(outBuf!, shape: [NSNumber(value: seqLen), NSNumber(value: dInner)], dataType: .float32)
//                
//                let yup = graph.multiplication(yPlaceholder, graph.silu(res), name:nil)
//                
//                
//                /// Projection of shape (l, 2(dInner)), where left is x and right is residual
//                let hmmmm = graph.matrixMultiplication(primary: yup,
//                                                       secondary: graph.transposeTensor(layer.outProj.key,
//                                                                                        dimension: 0,
//                                                                                        withDimension: 1,
//                                                                                        name: "mamba:output.transpose@\(layer.index)"),
//                                                       name: "mamba:output.matmul@\(layer.index)")
//                
//                let residual = graph.addition(hmmmm, xp, name: "residual:+@\(layer.index)")
//                let layerResults = graph.run(feeds: [yPlaceholder:yTensorData,
//                                                        xp:x
//                                                    ].merging(operands.flatMap({$0.feeds}), uniquingKeysWith: {$1}),
//                                             targetTensors: [residual],
//                                             targetOperations: nil)
//                x = layerResults[residual]!
//            }
//            
//            let vocEmbP = graph.placeholder(shape: state.embeddings.shape, dataType: .float32, name: "vocEmbP")
//            let normFPlaceholder  = graph.placeholder(shape: state.normF.shape, dataType: .float32, name: "normFP")
//            let laysors = graph.placeholder(shape: [NSNumber(value:tokens.count),NSNumber(value: model.dModel)], name: nil)
//            let normFoo = graph.rmsNorm(laysors, weights: normFPlaceholder)
//            let logits  = graph.matrixMultiplication(primary: normFoo,
//                                                     secondary: graph.transposeTensor(vocEmbP, dimension: 0, withDimension:1, name: "logits.transpose"),
//                                                     name: "logits.matmul")
//            let probs   = graph.softMax(with:
//                                        graph.sliceTensor(logits,
//                                                          dimension: 0,
//                                                          start: Int(truncating: logits.shape![0]) - 1,
//                                                          length: 1,
//                                                          name:nil),
//                                      axis: -1,
//                                      name: nil)
//            
//            print("Sampling...")
//            let results = graph.run(
//                feeds: [laysors:x,
//                        vocEmbP:state.embeddings,
//                        normFPlaceholder:state.normF,
//                       ].merging(operands.flatMap({$0.feeds}), uniquingKeysWith: {$1}),
//                targetTensors: [probs],
//                targetOperations: nil
//            )
//            
//            peekFloatTensorFloats(results[probs])
//            let arr = results[probs]!.mpsndarray()
//            let totalSize = (0..<arr.numberOfDimensions)
//                .reduce(into: 1, {$0 *= arr.length(ofDimension: $1)})
//            
//            
//            var values: [Float32] = .init(repeating: -42.0, count: totalSize)
//            arr.readBytes(&values, strideBytes: nil)
//            var tokenProbs: [(Int, Float32)] = Array(values.enumerated()
//                .map{ (i, v) in v.isNaN ? (i, 0) : (i, v) }
//                .sorted(by: {$0.1 > $1.1})
//                .prefix(40))
//            
//            
//            // Sum of all probabilities (so that we don't have to require that the sum is 1.0):
//            let sum = tokenProbs.reduce(0, {(a, v) in a + v.1})
//            // Random number in the range 0.0 <= rnd < sum :
//            let rnd = Float32.random(in: 0.0 ..< sum)
//            var choose = tokenProbs.count - 1
//            
//            // Find the first interval of accumulated probabilities into which `rnd` falls:
//            var accum: Float32 = 0.0
//            for (i, p) in tokenProbs.enumerated() {
//                accum += p.1
//                if rnd < accum {
//                    choose = i
//                    break
//                }
//            }
//            let choice = tokenProbs[choose].0
//            tokens = tokens + [Int32(choice)]
//            print(tokenizer.unTokenize(tokens))
//        }
        print("Done")
    }
    
    
    func genToselScan(_ graph: MambaMPSGraph, 
                      model: MambaRunner,
                      device: MTLDevice,
                      state: MambaState,
                      inTokens: [Int32],
                      layer: MambaBlockOperand,
                      x2: MPSGraphTensor,
                      res: MPSGraphTensor,
                      l: NSNumber,
                      xp: MPSGraphTensor,
                      x: MPSGraphTensorData,
                      margs: MambaArgs
    ) throws -> [MPSGraphTensorData] {
        
        let dConv: NSNumber = 4
        let dtRank: NSNumber = 48
        
        // x (l,dInner) -> (dInner, l)
        let trans = graph.transpose(x2, permutation: [1,0], name: "mamba:trans.transpose@\(layer.index)")
        
        // Conv Input: (N:1, C:dInner, H:1, W:L)
        let x2Panded = graph.reshape(trans, shape: [1, margs.dInner, 1, l], name: "mamba:x2Panded.reshape@\(layer.index)")
        
        // Conv Kernel: (N:1, C:dInner, H:1, W:dConv)
        let weXpand = graph.reshape(layer.conv1dWeight.key, shape: [1, margs.dInner, 1, dConv], name: "mamba:weXpand.reshape@\(layer.index)")
        
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
        let conv2d = graph.depthwiseConvolution2D(x2Panded,
                                            weights: weXpand,
                                            descriptor: descriptor!,
                                            name: "mamba:conv2d.depthwiseConvolution2D@\(layer.index)")
        // First of what is sure to be many Ls
        let firstL = graph.sliceTensor(conv2d, dimension: 3, start: 0, length: Int(truncating: l), name: "mamba:firstL.sliceTensor@\(layer.index)")
        let biased = graph.addition(firstL,
                                    graph.reshape(layer.conv1dBias.key, shape: [1, margs.dInner, 1, 1], name: "mamba:biased.reshape@\(layer.index)"),
                              name: "mamba:biased.addition@\(layer.index)")
        let transpose2 = graph.transposeTensor(biased, dimension: 0, withDimension: 3, name: "mamba:transpose2.transposeTensor@\(layer.index)")
        let silt = graph.silu(transpose2)
        
        let aLogP = layer.aLog.key
        guard aLogP.shape?[0] == margs.dInner, let n = aLogP.shape?[1] else {
            throw MambaError.invalidParameterShape("ssm/aLogP@\(layer.index)")
        }
        guard let l = silt.shape?[0] else {
            throw MambaError.invalidParameterShape("ssm/x@\(layer.index)")
        }
        
        /// Compute ∆ A B C D, the state space parameters.
        ///     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        ///     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        ///                                  and is why Mamba is called **selective** state spaces)
        let negExp = graph.exponent(with: aLogP, name: "ssm:negExp.exponent@\(layer.index)")
        let A = graph.negative(with: negExp, name: "ssm:A.negative@\(layer.index)")
        let D = layer.D.key
        
        let xDbl = graph.matrixMultiplication(primary: graph.reshape(silt, shape:[l,margs.dInner], name:"ssm:xDbl.reshape@\(layer.index)"),
                                        secondary: graph.transpose(layer.xProj.key, permutation: [1,0], name: "ssm:xDbl.transpose@\(layer.index)"),
                                        name: "ssm:xDbl.matmul@\(layer.index)")
        
        // (L, dtRank * 2*n)
        let dbc = graph.split(xDbl, splitSizes: [margs.dtRank, n, n], axis: -1, name: "ssm:dbc.split@\(layer.index)")
        
        guard let delta = try? dbc[0], let B = try? dbc[1], let C = try? dbc[2] else {
            throw MambaError.invalidParameterShape("ssm/delta@\(layer.index)")
        }
        
        let dtp = graph.matrixMultiplication(primary: delta,
                                             secondary: graph.transpose(layer.dtProjWeight.key, permutation: [1, 0], name: "ssm:dtp.transpose@\(layer.index)"),
                                             name: "ssm:dtp.matmul@\(layer.index)")
        
        let dtpwb = graph.addition(dtp, layer.dtProjBias.key, name: "ssm:dtpwb.addition@\(layer.index)")
        let spDelta = graph.softplus(dtpwb)
        

        let results = graph.run(
            feeds: [xp:x,
                   ].merging(layer.feeds, uniquingKeysWith: {$1}),
            targetTensors: [silt, spDelta, A, B, C, D, res],
            targetOperations: nil
        )
        
        
        
//        dumpTensorDataToFile(results[silt], "/Users/walker-ai/Desktop/SelectiveScanTestingData/silt.bin")
//        dumpTensorDataToFile(results[spDelta], "/Users/walker-ai/Desktop/SelectiveScanTestingData/spDelta.bin")
//        dumpTensorDataToFile(results[A], "/Users/walker-ai/Desktop/SelectiveScanTestingData/A.bin")
//        dumpTensorDataToFile(results[B], "/Users/walker-ai/Desktop/SelectiveScanTestingData/B.bin")
//        dumpTensorDataToFile(results[C], "/Users/walker-ai/Desktop/SelectiveScanTestingData/C.bin")
//        dumpTensorDataToFile(results[D], "/Users/walker-ai/Desktop/SelectiveScanTestingData/D.bin")
//        print("Yep")
        
//        var outs: [Float] = .init(repeating: -1919.0, count: mp.resourceSize() / MemoryLayout<Float32>.stride)
//        outs.withUnsafeMutableBufferPointer { op in
//            mp.readBytes(op.baseAddress!, strideBytes: nil)
//        }
//        var siluBuf: MTLBuffer = try outs.withUnsafeMutableBufferPointer { op in
//            guard let buf = device.makeBuffer(bytes: op.baseAddress!, length: mp.resourceSize()) else {
//                throw MambaError.failedToMakeMetalBuffer
//            }
//            return buf
//        }
//        let xqxqx = dumpBuffer(buffer: siluBuf, as: Float32.self)

        
//        peekFloatTensorFloats(results[ssm])
//        let arr = results[ssm]!.mpsndarray()
//        let totalSize = (0..<arr.numberOfDimensions)
//            .reduce(into: 1, {$0 *= arr.length(ofDimension: $1)})
//        
//        
//        var values: [Float32] = .init(repeating: -42.0, count: totalSize)
//        arr.readBytes(&values, strideBytes: nil)
//        values = values.map{ $0.isNaN ? 0 : $0 }

        return [results[silt]!, 
                results[spDelta]!,
                results[A]!,
                results[B]!,
                results[C]!,
                results[D]!,
                results[res]!] // ygdwygd
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
        // (l, dModel)
        let normFoo = graph.rmsNorm(laysors, weights: normFPlaceholder)
        // (l, nVocab)
        let logits  = graph.matrixMultiplication(primary: normFoo,
                                                secondary: graph.transposeTensor(embeddingsPlaceholder, dimension: 0, withDimension:1, name: "logits.transpose"),
                                                name: "logits.matmul")
        // (1, nVocab)
        let probs   = graph.softMax(with:
                                    graph.sliceTensor(logits,
                                                      dimension: 0,
                                                      start: Int(truncating: logits.shape![0]) - 1,
                                                      length: 1,
                                                      name:nil),
                                  axis: -1,
                                  name: nil)
        let topK = graph.topK(probs, k: 40, name: nil)
        
        // (1, nVocab)
        let aboveBar = graph.lessThan(probs, graph.reductionMinimum(with: topK[0], axis: 1, name: nil) , name: nil)
        // MPS select only supports tensor dimensions in range N[1-65536]D[1-16384]C[1-65536]H[1-16384]W[1-16384]
        // (nVocab, 1, 1)
        let cutP = graph.select(predicate: graph.reshape(aboveBar, shape: [NSNumber(value: margs.vocabSize), 1, 1], name: nil),
                                trueTensor: graph.reshape(probs,shape: [NSNumber(value: margs.vocabSize), 1, 1], name: nil),
                                falseTensor: graph.constant(0.0, shape: [NSNumber(value: margs.vocabSize), 1, 1], dataType: .float32),
                                name: nil)
        // (nVocab)
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
    
    func ssm(passing x: MPSGraphTensor, into layer: MambaBlockOperand, margs: MambaArgs, shortCircuit: Bool = false) throws -> MPSGraphTensor {
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
        
        if (shortCircuit) {
            return softplus
        }
        
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
    public let state: MamBufLoSoldier
    public let dModel: Int = 768
    public let nVocab: Int = 50280
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    public let library: MTLLibrary
    
    public init(state: MamBufLoSoldier, device: MTLDevice, cmdQ: MTLCommandQueue, library: MTLLibrary) {
        self.state = state
        self.commandQueue = cmdQ
        self.device = device
        self.library = library
    }
    
    public func scalarBuffer<T: Numeric>(_ scalar: T) throws -> MTLBuffer {
        var buf: MTLBuffer?
        try withUnsafeBytes(of: scalar) { sp in
            guard let ba = sp.baseAddress else {
                throw MambaError.invalidParameterShape("Scalar has no base address!?")
            }
            buf = self.device.makeBuffer(bytes: ba, length: MemoryLayout<UInt32>.stride)
        }
        guard let someB = buf else {
            throw MambaError.failedToMakeMetalBuffer
        }
        return someB
    }
    
    public func vectorBuffer<T: Numeric>(_ seq: any Sequence<T>) throws -> MTLBuffer {
        var arr = ContiguousArray(seq)
        let byteLength = MemoryLayout<T>.stride * arr.count
        return try arr.withUnsafeMutableBufferPointer { bp in
            guard let ba = bp.baseAddress else {
                throw MambaError.invalidParameterShape("Cannot make buffer from empty array")
            }
            guard let buf = self.device.makeBuffer(bytes: ba, length: byteLength) else {
                throw MambaError.failedToMakeMetalBuffer
            }
            return buf
        }
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
    
    public func embed(_ tokenIds: [Int32]) throws -> MTLBuffer {
        guard let embeddings = state.base["embedding.weight"]?.data else {
            throw MambaError.missingData("embedding")
        }
        let seqLen: UInt32 = UInt32(tokenIds.count)
        
        let nVb = try scalarBuffer(UInt32(nVocab))
        let dMb = try scalarBuffer(UInt32(dModel))
        let sLb = try scalarBuffer(seqLen)
        
        let tokBuf = try vectorBuffer(tokenIds.map{UInt32($0)})

        let embFn = library.makeFunction(name: "getEmbeddings")!
        let computePipelineState = try device.makeComputePipelineState(function: embFn)
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        let outBufSize = tokenIds.count * Int(dModel)
        let outByteCount = outBufSize * MemoryLayout<Float32>.stride
        let outBuf = device.makeBuffer(length: outByteCount)
        var output = [Float]()
        
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { throw MambaError.failedToMakeCommandEncoder }
        computeEncoder.setComputePipelineState(computePipelineState)
        
        computeEncoder.setBuffer(tokBuf,  offset: 0, index: 0)
        computeEncoder.setBuffer(outBuf,  offset: 0, index: 1)
        computeEncoder.setBuffer(embeddings,  offset: 0, index: 2)
        computeEncoder.setBuffer(sLb,  offset: 0, index: 3)
        computeEncoder.setBuffer(nVb,  offset: 0, index: 4)
        computeEncoder.setBuffer(dMb,  offset: 0, index: 5)
        
        let w = computePipelineState.threadExecutionWidth
        let h = computePipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSizeMake(w, h, 1)
        let threadsPerGrid = MTLSize(width: Int(dModel), height: Int(seqLen), depth: 1)
        
        computeEncoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        guard let ret = outBuf else {
            throw MambaError.failedToMakeMetalBuffer
        }
        return ret
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
