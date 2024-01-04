import XCTest
import Demo
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
import Accelerate

final class MambaTests: XCTestCase {
    var embBuf: MTLBuffer? = nil
    var device: MTLDevice? = nil
    var state: MambaState? = nil
    var graph: MambaMPSGraph? = nil
    var model: MambaRunner? = nil
    
    override func setUpWithError() throws {
        let folderUrl = "/Users/walker-ai/Desktop/ModelWeights/mamba130m-blowout"
        guard var dev = MTLCreateSystemDefaultDevice(),
        var q = dev.makeCommandQueue(),
        var cmdBuf = q.makeCommandBuffer()
        else {
            print("Failed to acquire Metal device")
            throw MambaError.failedToMakeCommandBuffer
        }
        
        self.device = dev
        let contents = try FileManager.default.contentsOfDirectory(atPath: folderUrl).filter({$0 != ".DS_Store"})
        
        
        let MSB = MambaStateBuilder()
        for cont in contents
        {
            let metadataPath = folderUrl + "/" + cont + "/metadata.json"
            let metadataJson = try String(contentsOfFile: metadataPath)
            let metadata = try JSONDecoder().decode(MambaLayerMeta.self, from: metadataJson.data(using: .utf8)!)
            
            let binDataPath = folderUrl + "/" + cont + "/weights.bin"
            let buffer = try loadBinaryAsMetalBuffer(binDataPath: binDataPath, device: device!, metadata: metadata)
            let tensorData = MPSGraphTensorData(buffer,
                                                shape: metadata.shape,
                                                dataType: .float32,
                                                rowBytes: MemoryLayout<Float32>.size * Int(exactly: metadata.shape.last!)!)

            
            switch(metadata.category)
            {
            case .lm_head:
                MSB.lmHead = tensorData
                print("Loaded LMHead: shape \(tensorData.shape)")
            case .norm_f:
                MSB.normF = tensorData
                embBuf = buffer
                print("Loaded normF: shape \(tensorData.shape)")
            case .embedding:
                MSB.embeddings = tensorData
                print("Wrote embeddings: shape \(tensorData.shape)")
            case .layers:
                let lb = MSB.getLayerBuilder(index: metadata.index!)
                try lb.addTensorData(tensorData, metadata)
                print("Loaded \(cont): shape \(tensorData.shape) -> \(previewFloatTensorBytes(tensorData, offset: 0, n: 2))...")
            }
        }
        
        state = try MSB.validated()
        graph = MambaMPSGraph()
        model = MambaRunner(device: &device!, cmdBuf: &cmdBuf)
    }

    func testEmbeddings() throws {
        let initialTokens: [Int32] = [33298, 1457] // Happy New
        let tokenEmbeddings = model!.embed(initialTokens, embeddings: &embBuf!)
        model!.cmdBuf.addCompletedHandler({ _ in
            XCTAssert({tokenEmbeddings.length(ofDimension: 0) == 768}())
            XCTAssert({tokenEmbeddings.length(ofDimension: 1) == 2}())
            let totalSize = (0..<tokenEmbeddings.numberOfDimensions)
                .reduce(into: 1, {$0 *= tokenEmbeddings.length(ofDimension: $1)})
            var values: [Float32] = .init(repeating: -42.0, count: totalSize)
            tokenEmbeddings.readBytes(&values, strideBytes: nil)
            XCTAssertEqual(values[0],  0.4832, accuracy: 0.001)
            XCTAssertEqual(values[1],  0.1379, accuracy: 0.001)
            XCTAssertEqual(values[2], -0.0377, accuracy: 0.001)
            XCTAssertEqual(values[768],  0.3195, accuracy: 0.001)
            XCTAssertEqual(values[769], -0.1531, accuracy: 0.001)
            XCTAssertEqual(values[770], -0.1651, accuracy: 0.001)
        })
        model!.cmdBuf.commit()
    }

    func testEmbeddingGraph() throws {
        guard let state = self.state,
        let device = self.device,
        let graph = self.graph else {
            throw fatalError("No state")
        }
        let embeddingsPlaceholder = graph.placeholder(shape: state.embeddings.shape, dataType: .float32, name: "embeddingsP")

        print("Gathering random embeddings from MPSGraph...")
        self.measure {
            let initialTokens: [Int32] = .init(repeating: 0, count: 100000).map { _ in
                Int32.random(in: 0..<Int32(model!.nVocab))
            }
            var broadTokens = initialTokens.map {
                Array<Int32>.init(repeating: $0, count: Int(truncating: state.embeddings.shape[1]))
            }.flatMap {$0}
            
            let tokenArray = MPSNDArray(device: device,
                                        descriptor: MPSNDArrayDescriptor(
                                            dataType: .int32,
                                            shape: [NSNumber(value: broadTokens.count / Int(truncating: state.embeddings.shape[1])),
                                                    state.embeddings.shape[1] ] ) )
            let tokenData = MPSGraphTensorData(tokenArray)
            
            broadTokens.withUnsafeMutableBytes { ptr in
                tokenArray.writeBytes(ptr.baseAddress!, strideBytes: nil)
            }

            
            let inTokensPlaceholder   = graph.placeholder(shape: tokenData.shape, dataType: .int32,   name: "inputTokensP")
            let emb     = graph.embeddingsOf(inTokensPlaceholder, within: embeddingsPlaceholder)
            let _ = graph.run(
                feeds: [inTokensPlaceholder:tokenData, embeddingsPlaceholder:state.embeddings],
                targetTensors: [emb],
                targetOperations: nil
            )
        }
    }
    
    func testEmbeddingvDSP() throws {
        guard let state = self.state else {
            fatalError("No state")
        }
        if (model!.nVocab / 16384 > 16) {
            fatalError("Too many tokens in vocab")
        }
        let textureCount   = Int(model!.nVocab / 16384)
        let vocabRemainder = model!.nVocab % 16384
        
        var fullTexture = MTLTextureDescriptor()
        fullTexture.usage = .pixelFormatView
        
        var embeddingsVec: [Float] = .init(repeating: -42.0, count: state.embeddings.shape.reduce(into: 1, {$0=$0*Int(truncating:$1)}))
        state.embeddings.mpsndarray().readBytes(&embeddingsVec, strideBytes: nil)

        print("Gathering random embeddings from vDSP...")
        self.measure {
            let initialTokens: [UInt] = .init(repeating: 0, count: 100000).map { _ in
                UInt.random(in: 0..<UInt(model!.nVocab))
            }
            let _: [Float] = initialTokens.reduce([], { arr, tokenId in
                let indices: [UInt] = Array(tokenId..<tokenId+UInt(model!.dModel))
                return arr + vDSP.gather(embeddingsVec, indices: indices)
            })
        }
    }

}
