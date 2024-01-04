import Foundation
import MetalPerformanceShadersGraph

public struct MambaArgs {
    let dConv: NSNumber = 4
    let dModel: NSNumber = 768
    let expand: NSNumber = 2
    let vocabSize: Int = 50280
    var dInner: NSNumber { NSNumber(value: Int(truncating: dModel) * Int(truncating: expand)) }
    var dtRank: NSNumber { NSNumber(value: Int(truncating: dModel) / 16) }
}

public struct MambaLayerMeta: Decodable
{
    public enum CatType: String, Codable { case lm_head, norm_f, layers, embedding }
    public enum RoleType: String, Codable { case norm, mixer }
    public enum KernelType: String, Codable { case conv1d, x_proj, dt_proj, A_log, out_proj, in_proj, D }
    public enum TypeType: String, Codable { case weight, bias, D, A_log }
    public var category: CatType
    public var index: Int?
    public var role: RoleType?
    public var kernel: KernelType?
    public var type: TypeType
    public var shape: [NSNumber]
    public var stride: [Int]
    
    enum CodingKeys : String, CodingKey {
        case category
        case index
        case role
        case kernel
        case type
        case shape
        case stride
    }
    public init(from decoder: Decoder) throws {
        let values  = try decoder.container(keyedBy: CodingKeys.self)
        category    = try values.decode(CatType.self, forKey: .category)
        index       = try Int(values.decode(String.self, forKey: .index))
        role        = try? values.decode(RoleType.self, forKey: .role)
        kernel      = try? values.decode(KernelType.self, forKey: .kernel)
        type        = try values.decode(TypeType.self, forKey: .type)
        shape       = try values.decode(Array<Int>.self, forKey: .shape).map({NSNumber(value: $0)})
        stride      = try values.decode(Array<Int>.self, forKey: .stride)
    }
}

public struct MambaBlockState
{
    typealias Weight = MPSGraphTensorData
    typealias WeightAndBias = (MPSGraphTensorData, MPSGraphTensorData)
    var index: Int
    var D: Weight
    var inProj: Weight
    var conv1d: WeightAndBias
    var xProj: Weight
    var dtProj: WeightAndBias
    var ALog: Weight
    var outProj: Weight
    var norm: Weight
}

public struct MambaState
{
    public typealias Weight = MPSGraphTensorData
    public var lmHead: Weight
    public var layers: [MambaBlockState]
    public var embeddings: Weight
    public var embBuf: MTLBuffer? = nil
    public var normF: Weight
    
}

public class MambaBlockStateBuilder: Equatable, Hashable
{
    public static func == (lhs: MambaBlockStateBuilder, rhs: MambaBlockStateBuilder) -> Bool {
        return lhs.layerNumber == rhs.layerNumber
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(layerNumber)
    }
    
    public typealias Weight = MPSGraphTensorData
    public typealias WeightAndBias = (MPSGraphTensorData?, MPSGraphTensorData?)
    var layerNumber: Int
    var D: Weight?
    var inProj: Weight?
    var conv1d: WeightAndBias = (nil, nil)
    var xProj: Weight?
    var dtProj: WeightAndBias = (nil, nil)
    var ALog: Weight?
    var outProj: Weight?
    var norm: Weight?
    public init(layerNumber: Int) {self.layerNumber = layerNumber}
    
    public func addTensorData(_ td: MPSGraphTensorData, _ metadata: MambaLayerMeta) throws {
        switch (metadata.role, metadata.kernel, metadata.type) {
        case (.norm, _, _):
            norm = td
            break
        case (.mixer, .D, _):
            D = td
            break
        case (.mixer, .A_log, _):
            ALog = td
            break
        case (.mixer, .in_proj, .weight):
            inProj = td
            break
        case (.mixer, .conv1d, .weight):
            conv1d.0 = td
            break
        case (.mixer, .conv1d, .bias):
            conv1d.1 = td
            break
        case (.mixer, .x_proj, .weight):
            xProj = td
            break
        case (.mixer, .dt_proj, .weight):
            dtProj.0 = td
            break
        case (.mixer, .dt_proj, .bias):
            dtProj.1 = td
            break
        case (.mixer, .out_proj, .weight):
            outProj = td
            break
        default:
            throw MambaError.unknownLayer
        }
    }
    
    public func validated() throws -> MambaBlockState {
        guard let D = D,
              let inProj = inProj,
              let conv1dWeight = conv1d.0,
              let conv1dBias = conv1d.1,
              let xProj = xProj,
              let dtProjWeight = dtProj.0,
              let dtProjBias = dtProj.1,
              let ALog = ALog,
              let outProj = outProj,
              let norm = norm else {
            throw MambaError.incompleteLayer
        }
        return MambaBlockState(index: layerNumber, D: D, inProj: inProj, conv1d: (conv1dWeight, conv1dBias), xProj: xProj, dtProj: (dtProjWeight, dtProjBias), ALog: ALog, outProj: outProj, norm: norm)
    }
}

public class MambaStateBuilder
{
    public typealias Weight = MPSGraphTensorData
    public var lmHead: Weight?
    private var layerBuilders: Set<MambaBlockStateBuilder> = []
    public var embeddings: Weight?
    public var normF: Weight?
    public init() {}
    
    public func getLayerBuilder(index:Int) -> MambaBlockStateBuilder {
        guard let found = layerBuilders.first(where: {$0.layerNumber == index}) 
        else {
            let new = MambaBlockStateBuilder(layerNumber: index)
            layerBuilders.insert(new)
            return new
        }
        return found
    }
    
    public func validated() throws -> MambaState {
        guard let lmHead = lmHead,
              let embeddings = embeddings,
              let normF = normF else {
            throw MambaError.missingEdgeLayers
        }
        var builtLayers: [MambaBlockState] = []
        for layerb in layerBuilders {
            let built = try layerb.validated()
            builtLayers.append(built)
        }
        return MambaState(lmHead: lmHead, 
                          layers: builtLayers.sorted(by: {$0.index < $1.index}),
                          embeddings: embeddings, normF: normF)
    }
}
