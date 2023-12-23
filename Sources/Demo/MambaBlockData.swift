import Foundation
import MetalPerformanceShadersGraph

public struct MambaBlockOperand {
    typealias Feed = (key: MPSGraphTensor, value: MPSGraphTensorData)
    let norm:           Feed
    let xProj:          Feed
    let dtProjWeight:   Feed
    let dtProjBias:     Feed
    let aLog:           Feed
    let D:              Feed
    let inProj:         Feed
    let outProj:        Feed
    let conv1dWeight:   Feed
    let conv1dBias:     Feed
    var feeds: [Dictionary<MPSGraphTensor, MPSGraphTensorData>.Element] {
        return [norm, xProj, dtProjWeight, dtProjBias, aLog, D, inProj, outProj, conv1dWeight, conv1dBias]
    }
    var shapedTypeFeeds: [Dictionary<MPSGraphTensor, MPSGraphShapedType>.Element] {
        return [norm, xProj, dtProjWeight, dtProjBias, aLog, D, inProj, outProj, conv1dWeight, conv1dBias]
            .map {(key: $0.key, value: MPSGraphShapedType(shape: $0.value.shape, dataType: $0.value.dataType))}
    }
    let index: Int
    init(state: MambaBlockState, graph: MambaMPSGraph) {
        index = state.index
        norm            = (key: graph.placeholder(shape: state.norm.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).normP"),
                           value: state.norm)
        xProj           = (key: graph.placeholder(shape: state.xProj.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).xProjP"),
                           value: state.xProj)
        dtProjWeight    = (key: graph.placeholder(shape: state.dtProj.0.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).dtProjWeightP"),
                           value: state.dtProj.0)
        dtProjBias      = (key: graph.placeholder(shape: state.dtProj.1.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).dtProjBiasP"),
                           value: state.dtProj.1)
        aLog            = (key: graph.placeholder(shape: state.ALog.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).aLogP"),
                           value: state.ALog)
        D               = (key: graph.placeholder(shape: state.D.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).DP"),
                           value: state.D)
        inProj          = (key: graph.placeholder(shape: state.inProj.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).inProjP"),
                           value: state.inProj)
        outProj         = (key: graph.placeholder(shape: state.outProj.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).outProjP"),
                           value: state.outProj)
        conv1dWeight    = (key: graph.placeholder(shape: state.conv1d.0.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).conv1dWeightP"),
                           value: state.conv1d.0)
        conv1dBias      = (key: graph.placeholder(shape: state.conv1d.1.shape,
                                                  dataType: .float32,
                                                  name: "layer\(index).conv1dBiasP"),
                           value: state.conv1d.1)
    }
}
