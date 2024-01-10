import Foundation
import MamBufLo

struct OG130HParams: MambaHParams
{
    let dState = 16
    let nVocab = 50280
    let dModel = 768
    let expand = 2
    let dConv = 4
    var dtRank: Int { dModel / 16 }
    var dInner: Int { dModel * expand }
}

struct OG130: ModelStateSpec
{
    let name = "mamba-130m"
    let nLayers = 24
    let hp = OG130HParams()
    var stateShapes: [String : [Int]]
    {[
        "embedding.weight":     [hp.nVocab, hp.dModel],
        "lm_head.weight":       [hp.nVocab, hp.dModel],
        "norm_f.weight":        [hp.dModel],
    ]}
    
    var perLayerStateShapes: [String : [Int]] 
    {[
        "norm.weight":          [hp.dModel],
        "mixer.A_log":          [hp.dInner, hp.dState],
        "mixer.conv1d.bias":    [hp.dInner],
        "mixer.conv1d.weight":  [hp.dInner, 1, hp.dConv],
        "mixer.D":              [hp.dInner],
        "mixer.dt_proj.bias":   [hp.dInner],
        "mixer.dt_proj.weight": [hp.dInner, hp.dtRank],
        "mixer.in_proj.weight": [hp.dInner * hp.expand, hp.dModel],
        "mixer.out_proj.weight":[hp.dModel, hp.dInner],
        "mixer.x_proj.weight":  [hp.dtRank + 2*hp.dState, hp.dInner],
    ]}
}




























//public protocol ModelStatesDescription {
//    var baseStates: [StateDescription]
//    var layerStates: [Int:StateDescription]
//    subscript(name: String) -> StateDescription {
//        get {
//            
//        }
//        set(newValue) {
//            
//        }
//    }
//    var shape: [StateDimension] { get }
//}
//
//public protocol LayerStateDescription {
//    var layerNumber: Int { get }
//    func add(_ metadata: InputMetadata, binDataPath: String)
//    func validated() throws -> MambaBlockState
//    func complete() throws -> MambaBlockState
//}
//
//public protocol ModelStateDescription {
//    var HParamObservations: [StateDimension] { get set }
//    var baseStates:  BaseStatesDescription
//    var layerStates: [Int:LayerStateDescription]
//    func add(_ metadata: InputMetadata, binDataPath: String) throws
//    func complete() throws -> MambaState
//}
//
//public class OGStateBuilder
//{
//    public var embeddings: Weight?
//    public var lmHead: Weight?
//    public var normF: Weight?
//    private var layerBuilders: Set<MambaBlockStateBuilder> = []
//    public init() {}
//    
//    public func addState(_ metadata: InputMetadata, binDataPath: String) {
//        
//    }
//    
//    public func getLayerBuilder(index:Int) -> MambaBlockStateBuilder {
//        guard let found = layerBuilders.first(where: {$0.layerNumber == index}) 
//        else {
//            let new = MambaBlockStateBuilder(layerNumber: index)
//            layerBuilders.insert(new)
//            return new
//        }
//        return found
//    }
//    
//    public func validated() throws -> MambaState {
//        guard let lmHead = lmHead,
//              let embeddings = embeddings,
//              let normF = normF else {
//            throw MambaError.missingEdgeLayers
//        }
//        var builtLayers: [MambaBlockState] = []
//        for layerb in layerBuilders {
//            let built = try layerb.validated()
//            builtLayers.append(built)
//        }
//        return MambaState(lmHead: lmHead, 
//                          layers: builtLayers.sorted(by: {$0.index < $1.index}),
//                          embeddings: embeddings, normF: normF)
//    }
//}
