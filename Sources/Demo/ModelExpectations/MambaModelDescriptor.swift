import Foundation
import MetalPerformanceShaders

public enum MambaMetaError: Error {
    case dimensionalError(String),
         shapeConflict(String),
         layerResPosError(String),
         layerTransformError(String),
         unknownLayer,
         invalidWeightClass,
        orderOfOperationsError(String)
}

public struct MambaModelArgs {
    let dConv: Int = 4
    let dModel: Int = 768
    let expand: Int = 2
    let vocabSize: Int = 50280
    var dInner: Int { dModel * expand }
    var dtRank: Int { dModel / 16 }
}

public class MambaMetaLoader {
    struct LayerMeta {
        var D:          MPSImageDescriptor? = nil
        var inProj:     MPSImageDescriptor? = nil
        var conv1dW:    MPSImageDescriptor? = nil
        var conv1dB:    MPSImageDescriptor? = nil
        var xProj:      MPSImageDescriptor? = nil
        var dtProjW:    MPSImageDescriptor? = nil
        var dtProjB:    MPSImageDescriptor? = nil
        var ALog:       MPSImageDescriptor? = nil
        var outProj:    MPSImageDescriptor? = nil
        var norm:       MPSImageDescriptor? = nil
    }
    
    var dConvObserved:  Int?
    var dStateObserved: Int?
    var dtRankObserved: Int?
    var dModelObserved: Int?
    var expandObserved: Int?
    var nVocabObserved: Int?
    
    
    var embedding:  MPSImageDescriptor? = nil
    var lmHead:     MPSImageDescriptor? = nil
    var normF:      MPSImageDescriptor? = nil
    var layers:     [Int:LayerMeta]     = [:]
    
    func conserveDimension(_ old: inout Int?, new: Int) throws {
        if let old = old, old != new {
            throw MambaMetaError.shapeConflict("Shape changed from \(old) to \(new)")
        }
        old = new
    }
    
    public init() { }
    
    func includingLMHead(_ metadata: InputMetadata) throws -> MambaMetaLoader {
        guard metadata.shape.count == 2 else {
            throw MambaMetaError.dimensionalError("LMHead metadata should be (nVocab, dModel)")
        }
        try conserveDimension(&nVocabObserved, new: metadata.shape[0])
        try conserveDimension(&dModelObserved, new: metadata.shape[1])
        lmHead = MPSImageDescriptor(channelFormat: .float32,
                                    width: nVocabObserved!,
                                    height: dModelObserved!,
                                    featureChannels: 1)
        return self
    }
    
    func includingEmbedding(_ metadata: InputMetadata) throws -> MambaMetaLoader {
        guard metadata.shape.count == 2 else {
            throw MambaMetaError.dimensionalError("Embedding metadata should be (nVocab, dModel)")
        }
        try conserveDimension(&nVocabObserved, new: metadata.shape[0])
        try conserveDimension(&dModelObserved, new: metadata.shape[1])
        embedding = MPSImageDescriptor(channelFormat: .float32,
                                    width: nVocabObserved!,
                                    height: dModelObserved!,
                                    featureChannels: 1)
        return self
    }
    
    func includingNormF(_ metadata: InputMetadata) throws -> MambaMetaLoader {
        guard metadata.shape.count == 1 else {
            throw MambaMetaError.dimensionalError("NormF metadata should be (dModel)")
        }
        try conserveDimension(&dModelObserved, new: metadata.shape[0])
        normF = MPSImageDescriptor(channelFormat: .float32,
                                   width: 1,
                                   height: dModelObserved!,
                                   featureChannels: 1)
        return self
    }
    
    func includingLayerNorm(_ metadata: InputMetadata) throws -> MambaMetaLoader {
        guard let idx = metadata.index else { throw MambaMetaError.unknownLayer }
        guard metadata.shape.count == 1 else {
            throw MambaMetaError.dimensionalError("Layer \(idx) norm should be (dModel)")
        }
        try conserveDimension(&dModelObserved, new: metadata.shape[0])
        layers[idx] = layers[idx] ?? LayerMeta()
        layers[idx]?.norm = MPSImageDescriptor(channelFormat: .float32,
                                               width: 1,
                                               height: dModelObserved!,
                                               featureChannels: 1)
        return self
    }
    
    func includingConvParams(_ metadata: InputMetadata) throws -> MambaMetaLoader {
        guard let idx = metadata.index else { throw MambaMetaError.unknownLayer }
        switch metadata.weightClass {
        case .weight:
            guard metadata.shape.count == 3 else {
                throw MambaMetaError.dimensionalError("Layer \(idx) conv weights should be (dInner, b, dConv)")
            }
            guard let dm = dModelObserved else {
                throw MambaMetaError.orderOfOperationsError("Load layers after base parameters")
            }
            
            try conserveDimension(&expandObserved, new: (metadata.shape[0] / dm))
            try conserveDimension(&dConvObserved, new: metadata.shape[2])
            
            layers[idx] = layers[idx] ?? LayerMeta()
            layers[idx]?.conv1dW = MPSImageDescriptor(channelFormat: .float32,
                                                      width: dModelObserved!,
                                                      height: dConvObserved!,
                                                      featureChannels: expandObserved!)
            return self
        case .bias:
            guard metadata.shape.count == 1 else {
                throw MambaMetaError.dimensionalError("Layer \(idx) conv bias should be (dInner)")
            }
            guard let dm = dModelObserved else {
                throw MambaMetaError.orderOfOperationsError("Load convolution parameters after base parameters")
            }
            try conserveDimension(&expandObserved, new: (metadata.shape[0] / dm))
            
            layers[idx] = layers[idx] ?? LayerMeta()
            layers[idx]?.conv1dB = MPSImageDescriptor(channelFormat: .float32,
                                                      width: dModelObserved!,
                                                      height: 1,
                                                      featureChannels: expandObserved!)
            return self
        default:
            throw MambaMetaError.invalidWeightClass
        }
    }
    
    func includingXProjParams(_ metadata: InputMetadata) throws -> MambaMetaLoader {
        guard let idx = metadata.index else { throw MambaMetaError.unknownLayer }
        guard let dtr = dtRankObserved, let dm = dModelObserved else {
            throw MambaMetaError.orderOfOperationsError("Load X proj params last")
        }
        guard metadata.shape.count == 2 else {
            throw MambaMetaError.dimensionalError("Layer \(idx) X projection should be ((2*dState) + dtRank, dInner)")
        }
        
        try conserveDimension(&expandObserved, new: metadata.shape[1] / dm)
        try conserveDimension(&dStateObserved, new: (metadata.shape[0] - dtr) / 2)
        layers[idx] = layers[idx] ?? LayerMeta()
        layers[idx]?.norm = MPSImageDescriptor(channelFormat: .float32,
                                               width: dModelObserved!,
                                               height: metadata.shape[1],
                                               featureChannels: expandObserved!)
        return self
    }
    
    public func including(_ metadata: InputMetadata) throws -> MambaMetaLoader {
        
        switch(metadata.modelPos) {
        case .embedding:
            return try includingEmbedding(metadata)
        case .norm_f:
            return try includingNormF(metadata)
        case .lm_head:
            return try includingLMHead(metadata)
        case .layers:
            switch(metadata.resPos) {
            case .norm:
                return try includingLayerNorm(metadata)
            case .mixer:
                switch(metadata.transform) {
                case .conv1d:
                    return try includingConvParams(metadata)
                case .x_proj:
                    return try includingXProjParams(metadata)
                case .dt_proj:
                    break
                case .A_log:
                    break
                case .out_proj:
                    break
                case .in_proj:
                    break
                case .D:
                    break
                case .none:
                    throw MambaMetaError.layerTransformError("Mixer should be one of [conv1d, x_proj, dt_proj, A_log, out_proj]")
                }
            case .none:
                throw MambaMetaError.layerResPosError("Layer should be one of [norm] or [mixer]")
            }
        }
        return self
    }
    
//    public func completing() -> MambaLoader {
//        return MambaLoader(args: MambaModelArgs())
//    }
}

public struct MambaLoader {
    let args: MambaModelArgs
    
}
