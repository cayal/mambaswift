import Foundation
import MetalPerformanceShadersGraph

func peekFloatTensorBytes(_ tensor: MPSGraphTensorData?, offset: Int = 0, n: Int = 32) {
    print(previewFloatTensorBytes(tensor, offset: offset, n: n))
}

func previewFloatTensorBytes(_ tensor: MPSGraphTensorData?, offset: Int = 0, n: Int = 32) -> String {
    guard let tensor = tensor else {
        return "peekFloatTensorBytes: no tensor"
    }
    let arr = tensor.mpsndarray()
    let totalSize = (0..<arr.numberOfDimensions)
        .reduce(into: 1, {$0 *= arr.length(ofDimension: $1)})
    
    guard tensor.dataType == .float32 else {
        return "peekFloatTensorBytes: expected float32, got \(tensor.dataType)"
    }
    
    var values: [Float32] = .init(repeating: -42.0, count: totalSize)
    arr.readBytes(&values, strideBytes: nil)
    var ie = offset
    var s = ""
    while ie < (offset + n) {
        s += String(format: "%08X", values[ie].bitPattern.bigEndian) + " "
        ie += 1
    }
    return s
}

func peekFloatTensorFloats(_ tensor: MPSGraphTensorData?, offset: Int = 0, nValues: Int = 8) {
    guard let tensor = tensor else {
        print("peekFloatTensorFloats: no tensor")
        return
    }
    let arr = tensor.mpsndarray()
    let totalSize = (0..<arr.numberOfDimensions)
        .reduce(into: 1, {$0 *= arr.length(ofDimension: $1)})
    
    guard tensor.dataType == .float32 else {
        print("peekFloatTensorBytes: expected float32, got \(tensor.dataType)")
        return
    }
    
    var values: [Float32] = .init(repeating: -42.0, count: totalSize)
    arr.readBytes(&values, strideBytes: nil)
    var ie = offset
    while ie < (offset + nValues) {
        guard ie < totalSize else { print("peekFloatTensorBytes: end of index!"); break }
        print(String(format: "%0.9f", values[ie]), terminator: " ")
        ie += 1
    }
    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(arr.label ?? "(peekFloatTensorBytes: no label)")
}
