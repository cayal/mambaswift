import Foundation
import Metal
import MetalPerformanceShaders

public struct InputMetadata: Decodable
{
    enum ModelPos: String, Codable { case lm_head, norm_f, layers, embedding }
    enum ResPos: String, Codable { case norm, mixer }
    enum Transform: String, Codable { case conv1d, x_proj, dt_proj, A_log, out_proj, in_proj, D }
    enum WeightClass: String, Codable { case weight, bias, D, A_log }
    var modelPos: ModelPos
    var index: Int?
    var resPos: ResPos?
    var transform: Transform?
    var weightClass: WeightClass
    var shape: [Int]
    var stride: [Int]
    
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
        modelPos    = try values.decode(ModelPos.self, forKey: .category)
        index       = try Int(values.decode(String.self, forKey: .index))
        resPos      = try? values.decode(ResPos.self, forKey: .role)
        transform   = try? values.decode(Transform.self, forKey: .kernel)
        weightClass = try values.decode(WeightClass.self, forKey: .type)
        shape       = try values.decode(Array<Int>.self, forKey: .shape)
        stride      = try values.decode(Array<Int>.self, forKey: .stride)
    }
}
