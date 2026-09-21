import Foundation
import Testing
@testable import SwiftletCore

@Suite struct GLMNextTests {
    static let fixtures = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("fixtures/glm-next")

    func root(_ layout: String = "stacked") throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtures.appendingPathComponent("\(layout)/config.json"))) as? [String: Any])
    }

    func copyFixture(_ layout: String = "stacked") throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("glm-\(UUID().uuidString)")
        try FileManager.default.copyItem(at: Self.fixtures.appendingPathComponent(layout), to: directory)
        return directory
    }

    func close(_ actual: [Float], _ expected: [Float], tolerance: Float = 0.00003) {
        #expect(actual.count == expected.count)
        guard actual.count == expected.count else { return }
        let error = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        #expect(error <= tolerance, "max absolute error \(error)")
    }

    @Test func configurationScheduleAndState() throws {
        let config = try GLMNextConfig(root: root())
        #expect(config.mlpLayerTypes == ["dense", "sparse"])
        #expect(config.layerTypes == ["linear_attention", "linear_attention"])
        #expect(config.recurrentStateBytes == 2 * 2 * (8 * 8 + 3 * 8 * 3) * 4)
        var value = try root()
        var text = try #require(value["text_config"] as? [String: Any])
        text["num_hidden_layers"] = 45
        text["first_k_dense_replace"] = 3
        text["linear_attn_config"] = ["num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4]
        value["text_config"] = text
        let real = try GLMNextConfig(root: value)
        #expect(real.layerTypes.filter { $0 == "deepseek_sparse_attention" }.count == 11)
        #expect(real.mlpLayerTypes.filter { $0 == "sparse" }.count == 42)
        #expect(real.recurrentStateBytes == 152_633_344)
    }

    @Test func rejectsInvalidArchitectureAndRouting() throws {
        for (key, value) in [("model_type", "qwen3_next" as Any), ("n_group", 3),
                             ("num_experts_per_tok", 9), ("scoring_func", "softmax"),
                             ("mlp_layer_types", ["sparse"]), ("first_k_dense_replace", -1)] {
            var cfg = try root()
            var text = try #require(cfg["text_config"] as? [String: Any])
            text[key] = value
            cfg["text_config"] = text
            #expect(throws: (any Swift.Error).self) { _ = try GLMNextConfig(root: cfg) }
        }
        #expect(throws: QwenConfig.Error.self) {
            _ = try QwenConfig(url: Self.fixtures.appendingPathComponent("stacked/config.json"))
        }
    }

    @Test(arguments: ["stacked", "split"])
    func headerOnlyAuditAndMixedPrecisionForward(_ layout: String) throws {
        let checkpoint = try GLMNextCheckpoint(directory: Self.fixtures.appendingPathComponent(layout))
        let audit = checkpoint.audit()
        #expect(checkpoint.weightBytesRead == 0)
        #expect(audit.routedExpertBytes == 77_824) // 8 x (4-bit + 4-bit + 5-bit + 3 scale/bias pairs)
        #expect(audit.storedBytes == audit.routedExpertBytes + audit.otherWeightBytes)
        #expect(audit.sparseLayerCount == 1)
        #expect(!audit.inferenceAvailable)
        let oracle = try SafetensorsFile(url: Self.fixtures.appendingPathComponent("reference.safetensors"))
        let inputs = try oracle.floats("input")
        let sparse = try oracle.floats("sparse_output")
        let dense = try oracle.floats("dense_output")
        for token in 0..<3 {
            let range = token * 64..<(token + 1) * 64
            close(try GLMNextMoE.forward(checkpoint: checkpoint, layer: 1, input: Array(inputs[range])), Array(sparse[range]))
            close(try GLMNextMoE.forward(checkpoint: checkpoint, layer: 0, input: Array(inputs[range])), Array(dense[range]))
        }
        #expect(checkpoint.peakReadBytes <= 6144, "must not fetch a full stacked expert tensor")
    }

    @Test func affineAllBitWidthsMatchMLX() throws {
        let checkpoint = try GLMNextCheckpoint(directory: Self.fixtures.appendingPathComponent("stacked"))
        let oracle = try SafetensorsFile(url: Self.fixtures.appendingPathComponent("reference.safetensors"))
        for bits in [2, 3, 4, 5, 6, 8] {
            for group in [32, 64, 128] {
                let module = "quant\(bits)g\(group)"
                close(try checkpoint.matrix(module, rows: 3, columns: 256), try oracle.floats(module), tolerance: 0)
            }
        }
    }

    @Test func orcaRouterStorageNamesAndStackedQuantOverrides() throws {
        let directory = try copyFixture("split")
        defer { try? FileManager.default.removeItem(at: directory) }
        // Match the real checkpoint's index/config layout, while keeping the
        // independent MLX numerical oracle and small synthetic weight bytes.
        func storedName(_ name: String) -> String {
            name.replacingOccurrences(of: "language_model.model.", with: "model.language_model.")
        }
        var cfg = try root("split")
        let oldQuant = try #require(cfg["quantization"] as? [String: Any])
        var quant: [String: Any] = [:]
        for (key, value) in oldQuant {
            let canonical = key.replacingOccurrences(of: "language_model.model.", with: "model.")
            let parts = canonical.split(separator: ".").map(String.init)
            if parts.count == 7, parts[4] == "experts" {
                quant[parts.prefix(4).joined(separator: ".") + ".switch_mlp." + parts[6]] = value
            } else if canonical.contains(".gate_up_proj") {
                quant[canonical.replacingOccurrences(of: ".gate_up_proj", with: ".gate_proj")] = value
                quant[canonical.replacingOccurrences(of: ".gate_up_proj", with: ".up_proj")] = value
            } else { quant[canonical] = value }
        }
        cfg["quantization"] = quant
        try JSONSerialization.data(withJSONObject: cfg).write(to: directory.appendingPathComponent("config.json"))
        var map: [String: String] = [:]
        for shard in 1...3 {
            let filename = String(format: "model-%05d-of-00003.safetensors", shard)
            let url = directory.appendingPathComponent(filename)
            var entries: [(name: String, dtype: String, shape: [Int], bytes: Data)] = []
            do {
                let file = try SafetensorsFile(url: url)
                for name in file.tensors.keys.sorted() {
                    let raw = try file.raw(name)
                    if name.contains(".gate_up_proj.") {
                        var shape = raw.info.shape
                        shape[0] /= 2
                        for (part, projection) in ["gate_proj", "up_proj"].enumerated() {
                            let renamed = storedName(name.replacingOccurrences(of: ".gate_up_proj.", with: ".\(projection)."))
                            let half = raw.bytes.count / 2
                            entries.append((renamed, raw.info.dtype, shape, raw.bytes.subdata(in: part * half..<(part + 1) * half)))
                            map[renamed] = filename
                        }
                    } else {
                        entries.append((storedName(name), raw.info.dtype, raw.info.shape, raw.bytes))
                        map[storedName(name)] = filename
                    }
                }
            }
            try SafetensorsFile.write(to: url, tensors: entries)
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": map]).write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        let checkpoint = try GLMNextCheckpoint(directory: directory)
        #expect(checkpoint.audit().routedExpertBytes == 77_824)
        #expect(checkpoint.weightBytesRead == 0)
        let oracle = try SafetensorsFile(url: Self.fixtures.appendingPathComponent("reference.safetensors"))
        let input = try oracle.floats("input")
        for (layer, tensor) in [(0, "dense_output"), (1, "sparse_output")] {
            let expected = try oracle.floats(tensor)
            for token in 0..<3 {
                let range = token * 64..<(token + 1) * 64
                close(try GLMNextMoE.forward(checkpoint: checkpoint, layer: layer, input: Array(input[range])), Array(expected[range]))
            }
        }
        let direct = try checkpoint.matrix("model.language_model.layers.1.mlp.experts.0.down_proj", rows: 64, columns: 64)
        close(direct, try checkpoint.expertProjection(layer: 1, expert: 0, projection: "down_proj"), tolerance: 0)
    }

    @Test func groupedRoutingMatchesUpstream() throws {
        let checkpoint = try GLMNextCheckpoint(directory: Self.fixtures.appendingPathComponent("stacked"))
        let oracle = try SafetensorsFile(url: Self.fixtures.appendingPathComponent("reference.safetensors"))
        let logits = try oracle.floats("logits")
        let bias = try checkpoint.vector("model.layers.1.mlp.gate.e_score_correction_bias", count: 8)
        let metadata = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtures.appendingPathComponent("reference.json"))) as? [String: Any])
        let cases = try #require(metadata["routing_cases"] as? [[String: Any]])
        for test in cases {
            var cfg = try root()
            var text = try #require(cfg["text_config"] as? [String: Any])
            text["num_experts_per_tok"] = test["top_k"]
            text["n_group"] = test["groups"]
            text["topk_group"] = test["kept"]
            text["norm_topk_prob"] = test["normalize"]
            cfg["text_config"] = text
            let config = try GLMNextConfig(root: cfg)
            let ids = try #require(test["ids"] as? [[Int]])
            let weights = try #require(test["weights"] as? [[Double]])
            for token in 0..<3 {
                let routes = try GLMNextMoE.route(logits: Array(logits[token * 8..<(token + 1) * 8]), correctionBias: bias, config: config)
                #expect(Set(routes.map(\.expert)) == Set(ids[token]))
                for route in routes {
                    let i = try #require(ids[token].firstIndex(of: route.expert))
                    #expect(abs(route.weight - Float(weights[token][i])) < 0.000001)
                }
            }
        }
    }

    @Test func selectionReadsOnlyRequestedExpert() throws {
        let checkpoint = try GLMNextCheckpoint(directory: Self.fixtures.appendingPathComponent("stacked"))
        _ = try checkpoint.expertProjection(layer: 1, expert: 7, projection: "down_proj")
        #expect(checkpoint.weightBytesRead == 64 * 64 * 5 / 8 + 2 * 64 * 2 * 4)
        #expect(throws: (any Swift.Error).self) { _ = try checkpoint.expertProjection(layer: 0, expert: 0, projection: "up_proj") }
        #expect(throws: (any Swift.Error).self) { _ = try checkpoint.expertProjection(layer: 1, expert: 8, projection: "up_proj") }
        #expect(throws: (any Swift.Error).self) { _ = try checkpoint.matrix("quant4g32", rows: 3, columns: 255) }
        #expect(throws: (any Swift.Error).self) {
            _ = try GLMNextMoE.forward(checkpoint: checkpoint, layer: 1, input: [.nan])
        }
    }

    @Test func rejectsMissingShardAndIndexMismatch() throws {
        let directory = try copyFixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let index = directory.appendingPathComponent("model.safetensors.index.json")
        var object = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any])
        var map = try #require(object["weight_map"] as? [String: String])
        map["missing.weight"] = "model-00001-of-00003.safetensors"
        object["weight_map"] = map
        try JSONSerialization.data(withJSONObject: object).write(to: index)
        #expect(throws: (any Swift.Error).self) { _ = try GLMNextCheckpoint(directory: directory) }
        map["missing.weight"] = "../outside.safetensors"
        object["weight_map"] = map
        try JSONSerialization.data(withJSONObject: object).write(to: index)
        #expect(throws: (any Swift.Error).self) { _ = try GLMNextCheckpoint(directory: directory) }
    }

    @Test func rejectsTruncatedShardBeforeReadingWeights() throws {
        let directory = try copyFixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model-00001-of-00003.safetensors")
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: 16)
        try handle.close()
        #expect(throws: (any Swift.Error).self) { _ = try GLMNextCheckpoint(directory: directory) }
    }

    @Test func inventoriesScalarAuxiliariesWithoutReadingWeights() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("glm-scalar-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: root()).write(to: directory.appendingPathComponent("config.json"))
        var scalar: Float = 1
        let data = withUnsafeBytes(of: &scalar) { Data($0) }
        try SafetensorsFile.write(to: directory.appendingPathComponent("model.safetensors"),
                                 tensors: [(name: "auxiliary_scale", dtype: "F32", shape: [], bytes: data)])
        let checkpoint = try GLMNextCheckpoint(directory: directory)
        #expect(checkpoint.audit().otherWeightBytes == 4)
        #expect(checkpoint.weightBytesRead == 0)
    }
}
