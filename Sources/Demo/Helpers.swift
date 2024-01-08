import Foundation
import MetalPerformanceShadersGraph

public func peekFloatTensorBytes(_ tensor: MPSGraphTensorData?, offset: Int = 0, n: Int = 32) {
    print(previewFloatTensorBytes(tensor, offset: offset, n: n))
}

public func previewFloatTensorBytes(_ tensor: MPSGraphTensorData?, offset: Int = 0, n: Int = 32) -> String {
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

public func peekFloatTensorFloats(_ tensor: MPSGraphTensorData?, offset: Int = 0, nValues: Int = 8) {
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

public func peekFloatNDFloats(_ nd: MPSNDArray?, offset: Int = 0, nValues: Int = 8) {
    guard let arr = nd else {
        print("peekFloatNDFloats: no NDArray")
        return
    }
    let totalSize = (0..<arr.numberOfDimensions)
        .reduce(into: 1, {$0 *= arr.length(ofDimension: $1)})
    
    var values: [Float32] = .init(repeating: -42.0, count: totalSize)
    arr.readBytes(&values, strideBytes: nil)
    var ie = offset
    while ie < (offset + nValues) {
        guard ie < totalSize else { print("peekFloatNDFloats: end of index!"); break }
        print(String(format: "%0.9f", values[ie]), terminator: " ")
        ie += 1
    }
    print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print(arr.label ?? "(peekFloatNDFloats: no label)")
}

public func peekNDBytes(_ nd: MPSNDArray?, offset: Int = 0, n: Int = 8) {
    print(previewNDBytes(nd, offset: offset, nValues: n))
}

public func previewNDBytes(_ nd: MPSNDArray?, offset: Int = 0, nValues: Int = 8) -> String {
    guard let arr = nd else {
        print("peekFloatNDFloats: no NDArray")
        return ""
    }
    let totalSize = (0..<arr.numberOfDimensions)
        .reduce(into: 1, {$0 *= arr.length(ofDimension: $1)})
    
    var values: [Int32] = .init(repeating: -42, count: totalSize)
    arr.readBytes(&values, strideBytes: nil)
    var ie = offset
    var s = ""
    while ie < (offset + nValues) {
        guard ie < totalSize else { print("previewNDBytes: end of index!"); break }
        s += String(format: "%08X", values[ie].bigEndian) + " "
        ie += 1
    }
    return s
}


public func loadBinaryAsMetalBuffer(binDataPath: String, device: MTLDevice, metadata: MambaLayerMeta) throws -> MTLBuffer {
    let url = URL(fileURLWithPath: binDataPath)
    let fileHandle = try FileHandle(forReadingFrom: url)
    guard let fileAttributes = try? FileManager.default.attributesOfItem(atPath: url.path()) else {
        throw MambaError.invalidFile(url.absoluteString)
    }
    
    var data = try fileHandle.readToEnd()
    defer {
        fileHandle.closeFile()
    }
    
    var strideBytes = metadata.stride.map {$0 * MemoryLayout<Float32>.stride }
    guard let dataSize = data?.count,
          dataSize == metadata.shape.reduce(MemoryLayout<Float32>.size, {$0 * Int($1)}) else {
        throw MambaError.incompleteLayer
    }
    var buf: MTLBuffer? = nil
    data!.withUnsafeMutableBytes({ (ptr: UnsafeMutableRawBufferPointer) -> Void in
        buf = device.makeBuffer(bytesNoCopy: ptr.baseAddress!, length: dataSize, options: [.storageModeShared])
    })
    
    guard let someBuf = buf else {
        throw MambaError.failedToMakeMetalBuffer
    }
    return someBuf
}

public func loadBinaryAsNDArray(binDataPath: String, device: MTLDevice, metadata: MambaLayerMeta) throws -> MPSNDArray {
    let array = MPSNDArray(device: device, descriptor: MPSNDArrayDescriptor(dataType: .float32, shape: metadata.shape))
    let url = URL(fileURLWithPath: binDataPath)
    let fileHandle = try FileHandle(forReadingFrom: url)
    guard let fileAttributes = try? FileManager.default.attributesOfItem(atPath: url.path()),
          let fileSize = fileAttributes[.size] as? Int else {
        throw MambaError.invalidFile(url.absoluteString)
    }
    
    var data = try fileHandle.readToEnd()
    defer {
        fileHandle.closeFile()
    }
    
    if(metadata.index == 7) {
        print("Unlucky 7!")
    }
    
    var strideBytes = metadata.stride.map {$0 * MemoryLayout<Float32>.stride }
    data!.withUnsafeMutableBytes({ (ptr: UnsafeMutableRawBufferPointer) -> Void in
        array.writeBytes(ptr.baseAddress!, strideBytes: &strideBytes)
        print("Array size \(array.resourceSize())")
    })
    return array
}

func dumpBuffer<T>(buffer: MTLBuffer, as type: T.Type) -> [T] {
    let outBufSize = buffer.length / MemoryLayout<T>.stride
    let dataPointer = buffer.contents().bindMemory(to: type, capacity: outBufSize)
    let bufferPointer = UnsafeBufferPointer(start: dataPointer, count: outBufSize)
    var output = [T]()
    bufferPointer.forEach { value in output.append(value) }
    return output
}

func dumpToFile<T>(array: [T], toFilePath path: String) {
    array.withUnsafeBufferPointer { bp in
        let data = Data(buffer: bp)
        do {
            try data.write(to: URL(fileURLWithPath: path))
            print("Data written to file successfully.")
        } catch {
            print("Error writing data to file: \(error)")
        }
    }
}

func dumpTensorDataToFile(_ tensorData: MPSGraphTensorData?, _ path: String) {
    guard let tensorData = tensorData else {
        print("dumpTensorDataToFile: TensorData is nil")
        return
    }
    let ndArray = tensorData.mpsndarray()
    let totalCount = tensorData.shape.reduce(1, {$0 * Int(truncating: $1)})
    let totalBytes = totalCount * MemoryLayout<Float32>.stride
    var arr: [Float32] = .init(repeating: -99.99, count: totalCount)
    let data: Data = arr.withUnsafeMutableBufferPointer { ap in
        ndArray.readBytes(ap.baseAddress!, strideBytes: nil)
        return Data(bytesNoCopy: ap.baseAddress!, count: totalBytes, deallocator: .none)
    }

    // Write Data to file
    do {
        try data.write(to: URL(fileURLWithPath: path))
        print("Data written to file successfully.")
    } catch {
        print("Error writing data to file: \(error)")
    }
}

                    
func withDebugCapture<T>(on device: MTLDevice, execute closure: () throws -> T) rethrows -> T {
    let sharedCapturer = MTLCaptureManager.shared()
    let customScope = sharedCapturer.makeCaptureScope(device: device)
    customScope.label = "Pls fix it"
    sharedCapturer.defaultCaptureScope = customScope

    let captureDescriptor = MTLCaptureDescriptor()
    captureDescriptor.captureObject = device
    do {
        try sharedCapturer.startCapture(with: captureDescriptor)
    } catch {
        fatalError("Failed to capture: \(error)")
    }
    customScope.begin()

    let result = try closure()

    customScope.end()
    sharedCapturer.stopCapture()

    return result
}
