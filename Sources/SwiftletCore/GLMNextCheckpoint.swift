import Foundation
import Darwin

/// Experimental GLM checkpoint reader. Opens only shard headers at startup;
/// selected matrix rows are fetched with pread, never mapping entire shards.
/// Supports split experts and MLX's stacked switch_mlp representation.
public final class GLMNextCheckpoint {
    public let config: GLMNextConfig
    private let quant: (default: Checkpoint.QuantSpec?, overrides: [String: Checkpoint.QuantSpec])
    private var shards: [Shard] = []
    private var tensors: [String: Tensor] = [:]
    public private(set) var weightBytesRead = 0
    public private(set) var peakReadBytes = 0

    private struct Tensor {
        let shard: Int
        let dtype: String
        let shape: [Int]
        let offset: Int
        let size: Int
    }
    private final class Shard {
        let handle: FileHandle
        init(_ url: URL) throws { handle = try FileHandle(forReadingFrom: url) }
        deinit { try? handle.close() }
    }

    public struct Audit: Codable, Sendable {
        public let storedBytes: Int
        public let routedExpertBytes: Int
        public let otherWeightBytes: Int
        public let recurrentStateBytes: Int
        public let sparseLayerCount: Int
        public let tensorCount: Int
        public let inferenceAvailable: Bool
        public let caveat: String
    }

    public init(directory: URL) throws {
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard let root = root as? [String: Any] else { throw GLMNextConfig.Error.invalid("invalid config object") }
        config = try GLMNextConfig(root: root)
        quant = try Checkpoint.quantization(fromConfig: root)
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        var weightMap: [String: String]?
        if FileManager.default.fileExists(atPath: indexURL.path) {
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any]
            guard let map = index?["weight_map"] as? [String: String], !map.isEmpty else {
                throw GLMNextConfig.Error.invalid("invalid safetensors weight map")
            }
            weightMap = map
        }
        let filenames = weightMap.map { Set($0.values).sorted() } ?? ["model.safetensors"]
        for filename in filenames {
            guard !filename.isEmpty, filename == (filename as NSString).lastPathComponent,
                  filename.hasSuffix(".safetensors") else {
                throw GLMNextConfig.Error.invalid("invalid shard filename \(filename)")
            }
            let shard = try Shard(directory.appendingPathComponent(filename))
            let fileSize = try shard.handle.seekToEnd()
            let lengthBytes = try Self.read(shard, offset: 0, count: 8)
            let length = lengthBytes.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            // Bound metadata allocations and reject arithmetic overflow before reads.
            guard fileSize >= 8, length <= 64 * 1024 * 1024, length <= fileSize - 8 else {
                throw GLMNextConfig.Error.invalid("invalid safetensors header length")
            }
            let start = 8 + Int(length)
            let header = try JSONSerialization.jsonObject(with: Self.read(shard, offset: 8, count: Int(length)))
            guard let entries = header as? [String: Any] else { throw GLMNextConfig.Error.invalid("invalid shard header") }
            var ranges: [Range<Int>] = []
            for (name, value) in entries where name != "__metadata__" {
                guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                      let elementSize = SafetensorsFile.bytesPerElement(dtype),
                      let shape = entry["shape"] as? [Int],
                      let offsets = entry["data_offsets"] as? [Int], offsets.count == 2,
                      offsets[0] >= 0, offsets[1] >= offsets[0],
                      UInt64(offsets[1]) <= fileSize - UInt64(start), tensors[name] == nil else {
                    throw GLMNextConfig.Error.invalid("invalid tensor metadata: \(name)")
                }
                let size = try Self.product(shape + [elementSize])
                guard size == offsets[1] - offsets[0] else {
                    throw GLMNextConfig.Error.invalid("shape/byte count mismatch: \(name)")
                }
                if let weightMap, weightMap[name] != filename {
                    throw GLMNextConfig.Error.invalid("index/header mismatch: \(name)")
                }
                ranges.append(offsets[0]..<offsets[1])
                tensors[name] = Tensor(shard: shards.count, dtype: dtype, shape: shape,
                                       offset: start + offsets[0], size: size)
            }
            let sorted = ranges.sorted { $0.lowerBound < $1.lowerBound }
            for i in sorted.indices.dropFirst() where sorted[i].lowerBound < sorted[i - 1].upperBound {
                throw GLMNextConfig.Error.invalid("overlapping safetensors data")
            }
            shards.append(shard)
        }
        if let weightMap, Set(weightMap.keys) != Set(tensors.keys) {
            throw GLMNextConfig.Error.invalid("weight map references absent tensors")
        }
    }

    private static func product(_ dimensions: [Int]) throws -> Int {
        var value = 1
        for dimension in dimensions {
            let (next, overflow) = value.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !overflow else { throw GLMNextConfig.Error.invalid("invalid tensor dimensions") }
            value = next
        }
        return value
    }

    private static func read(_ shard: Shard, offset: Int, count: Int) throws -> Data {
        var bytes = Data(count: count)
        try bytes.withUnsafeMutableBytes { raw in
            var done = 0
            while done < count {
                let n = pread(shard.handle.fileDescriptor, raw.baseAddress!.advanced(by: done), count - done, off_t(offset + done))
                if n < 0 && errno == EINTR { continue }
                guard n > 0 else { throw GLMNextConfig.Error.invalid("short/failed shard read at \(offset + done)") }
                done += n
            }
        }
        return bytes
    }

    private func resolve(_ name: String) -> String {
        if tensors[name] != nil { return name }
        return "language_model." + name
    }
    public func contains(_ name: String) -> Bool { tensors[resolve(name)] != nil }

    private func info(_ name: String) throws -> Tensor {
        guard let tensor = tensors[resolve(name)] else { throw Checkpoint.Error.missingTensor(name) }
        return tensor
    }

    private func bytes(_ name: String, range: Range<Int>) throws -> Data {
        let tensor = try info(name)
        guard range.lowerBound >= 0, range.upperBound <= tensor.size else {
            throw GLMNextConfig.Error.invalid("tensor read out of bounds: \(name)")
        }
        let result = try Self.read(shards[tensor.shard], offset: tensor.offset + range.lowerBound, count: range.count)
        weightBytesRead += result.count
        peakReadBytes = max(peakReadBytes, result.count)
        return result
    }

    private func floats(_ name: String, elements: Range<Int>) throws -> [Float] {
        let tensor = try info(name)
        guard ["F32", "F16", "BF16"].contains(tensor.dtype),
              let width = SafetensorsFile.bytesPerElement(tensor.dtype),
              elements.lowerBound >= 0, elements.upperBound <= tensor.size / width else {
            throw GLMNextConfig.Error.invalid("invalid float tensor/range: \(name)")
        }
        let data = try bytes(name, range: elements.lowerBound * width..<elements.upperBound * width)
        return data.withUnsafeBytes { raw in
            (0..<elements.count).map { i in
                switch tensor.dtype {
                case "F32": return Float(bitPattern: raw.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self).littleEndian)
                case "F16": return Float(Float16(bitPattern: raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self).littleEndian))
                default: return Float(bitPattern: UInt32(raw.loadUnaligned(fromByteOffset: i * 2, as: UInt16.self).littleEndian) << 16)
                }
            }
        }
    }

    public func vector(_ name: String, count: Int) throws -> [Float] {
        guard try info(name).shape == [count] else { throw Checkpoint.Error.badShape(name) }
        return try floats(name, elements: 0..<count)
    }

    /// Read one expert (or an ordinary matrix), preserving each projection's
    /// own affine precision. Bit fields may cross uint32 boundaries at 3/5/6 bits.
    public func matrix(_ module: String, rows: Int, columns: Int, expert: Int? = nil) throws -> [Float] {
        let weightName = module + ".weight"
        let weight = try info(weightName)
        let leading = expert == nil ? [rows] : [config.expertCount, rows]
        guard rows > 0, columns > 0, weight.shape.dropLast().elementsEqual(leading),
              expert == nil || (0..<config.expertCount).contains(expert!) else {
            throw Checkpoint.Error.badShape(weightName)
        }
        let rowStart = try Self.product([rows, (expert ?? 0) + 1]) - rows
        let scaleName = module + ".scales"
        if !contains(scaleName) {
            guard weight.shape.last == columns else { throw Checkpoint.Error.badShape(weightName) }
            let end = try Self.product([rowStart + rows, columns])
            return try floats(weightName, elements: rowStart * columns..<end)
        }
        // Only strip the known outer prefix; never suffix-match unrelated modules.
        let canonical = module.hasPrefix("language_model.") ? String(module.dropFirst(15)) : module
        guard let spec = quant.overrides[module] ?? quant.overrides["language_model." + canonical]
                ?? quant.overrides[canonical] ?? quant.default,
              [2, 3, 4, 5, 6, 8].contains(spec.bits), [32, 64, 128].contains(spec.groupSize),
              columns % spec.groupSize == 0, weight.dtype == "U32" else {
            throw GLMNextConfig.Error.invalid("unsupported or missing affine quantization: \(module)")
        }
        let rowBits = try Self.product([columns, spec.bits])
        let groups = columns / spec.groupSize
        let packedCols = rowBits / 32
        let expected = leading + [groups]
        guard rowBits % 32 == 0, weight.shape.last == packedCols,
              try info(scaleName).shape == expected, try info(module + ".biases").shape == expected else {
            throw Checkpoint.Error.badShape(module)
        }
        let byteEnd = try Self.product([rowStart + rows, packedCols, 4])
        let packed = try bytes(weightName, range: rowStart * packedCols * 4..<byteEnd)
        let scales = try floats(scaleName, elements: rowStart * groups..<(rowStart + rows) * groups)
        let biases = try floats(module + ".biases", elements: rowStart * groups..<(rowStart + rows) * groups)
        var result = [Float](repeating: 0, count: try Self.product([rows, columns]))
        let mask = UInt16((1 << spec.bits) - 1)
        packed.withUnsafeBytes { (raw: UnsafeRawBufferPointer) in
            for row in 0..<rows {
                for column in 0..<columns {
                    let bit = column * spec.bits
                    let offset = row * packedCols * 4 + bit / 8
                    var field = UInt16(raw[offset])
                    if bit % 8 + spec.bits > 8 { field |= UInt16(raw[offset + 1]) << 8 }
                    let value = Float((field >> (bit % 8)) & mask)
                    let group = row * groups + column / spec.groupSize
                    result[row * columns + column] = scales[group] * value + biases[group]
                }
            }
        }
        return result
    }

    public func expertProjection(layer: Int, expert: Int, projection: String) throws -> [Float] {
        guard (0..<config.layerCount).contains(layer), config.mlpLayerTypes[layer] == "sparse",
              (0..<config.expertCount).contains(expert), ["gate_proj", "up_proj", "down_proj"].contains(projection) else {
            throw GLMNextConfig.Error.invalid("invalid routed expert address")
        }
        let down = projection == "down_proj"
        let rows = down ? config.hiddenSize : config.moeIntermediateSize
        let columns = down ? config.moeIntermediateSize : config.hiddenSize
        let prefix = "model.layers.\(layer).mlp."
        let stacked = prefix + "switch_mlp." + projection
        if contains(stacked + ".weight") {
            return try matrix(stacked, rows: rows, columns: columns, expert: expert)
        }
        return try matrix(prefix + "experts.\(expert)." + projection, rows: rows, columns: columns)
    }

    /// Storage accounting only; otherWeightBytes is NOT a peak-RAM estimate.
    public func audit() -> Audit {
        var routed = 0
        var total = 0
        for (name, tensor) in tensors {
            total += tensor.size
            let canonical = name.hasPrefix("language_model.") ? String(name.dropFirst(15)) : name
            let parts = canonical.split(separator: ".")
            if parts.count >= 6, parts[0] == "model", parts[1] == "layers",
               let layer = Int(parts[2]), (0..<config.layerCount).contains(layer),
               config.mlpLayerTypes[layer] == "sparse", parts[3] == "mlp",
               parts[4] == "experts" || parts[4] == "switch_mlp" {
                routed += tensor.size
            }
        }
        return Audit(storedBytes: total, routedExpertBytes: routed, otherWeightBytes: total - routed,
                     recurrentStateBytes: config.recurrentStateBytes,
                     sparseLayerCount: config.mlpLayerTypes.filter { $0 == "sparse" }.count,
                     tensorCount: tensors.count, inferenceAvailable: false,
                     caveat: "Stored bytes, not a RAM-fit guarantee. Excludes attention caches, activations, expert cache, conversion/allocator overhead and OS memory. GLM attention and full generation are not implemented.")
    }
}
