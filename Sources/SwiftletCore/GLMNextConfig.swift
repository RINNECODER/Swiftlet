import Foundation

/// Configuration for the experimental GLM-5.3-Flash text/MoE port.
/// This is deliberately separate from QwenConfig: the two forward passes
/// are not interchangeable. Parsing this configuration does not enable chat.
public struct GLMNextConfig: Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let layerCount: Int
    public let expertCount: Int
    public let sharedExpertCount: Int
    public let topK: Int
    public let groupCount: Int
    public let topKGroups: Int
    public let normalizeTopK: Bool
    public let routedScalingFactor: Float
    public let swigluLimit: Float
    public let layerTypes: [String]
    public let mlpLayerTypes: [String]
    public let linearHeads: Int
    public let linearHeadDim: Int
    public let linearConvKernel: Int

    public enum Error: Swift.Error, CustomStringConvertible {
        case invalid(String)
        public var description: String {
            switch self { case .invalid(let message): return "GLM-Next: \(message)" }
        }
    }

    public init(url: URL) throws {
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw Error.invalid("config.json must be an object")
        }
        try self.init(root: root)
    }

    public init(root: [String: Any]) throws {
        let text = root["text_config"] as? [String: Any] ?? root
        guard ["glm5_next", "glm5_next_text"].contains(text["model_type"] as? String ?? "") else {
            throw Error.invalid("expected glm5_next or glm5_next_text model_type")
        }
        func integer(_ name: String, default fallback: Int? = nil) throws -> Int {
            guard let value = text[name] as? Int ?? fallback, value > 0, value <= 1_048_576 else {
                throw Error.invalid("invalid or missing \(name)")
            }
            return value
        }
        hiddenSize = try integer("hidden_size")
        intermediateSize = try integer("intermediate_size")
        moeIntermediateSize = try integer("moe_intermediate_size")
        layerCount = try integer("num_hidden_layers")
        expertCount = try integer("n_routed_experts")
        sharedExpertCount = try integer("n_shared_experts", default: 1)
        topK = try integer("num_experts_per_tok")
        groupCount = try integer("n_group", default: 1)
        topKGroups = try integer("topk_group", default: 1)
        guard expertCount % groupCount == 0, topKGroups <= groupCount,
              topK <= expertCount / groupCount * topKGroups,
              groupCount == 1 || expertCount / groupCount >= 2 else {
            throw Error.invalid("inconsistent expert/group/top-k counts")
        }
        guard (text["scoring_func"] as? String ?? "sigmoid") == "sigmoid",
              (text["topk_method"] as? String ?? "noaux_tc") == "noaux_tc",
              (text["hidden_act"] as? String ?? "silu") == "silu" else {
            throw Error.invalid("only sigmoid/noaux_tc routing and silu are implemented")
        }
        normalizeTopK = text["norm_topk_prob"] as? Bool ?? true
        routedScalingFactor = (text["routed_scaling_factor"] as? NSNumber)?.floatValue ?? 2.5
        swigluLimit = (text["swiglu_limit"] as? NSNumber)?.floatValue ?? 10
        guard routedScalingFactor.isFinite, routedScalingFactor > 0,
              swigluLimit.isFinite, swigluLimit > 0 else {
            throw Error.invalid("invalid routing scale or SwiGLU limit")
        }
        let dense = text["first_k_dense_replace"] as? Int ?? 3
        guard dense >= 0, dense <= layerCount else { throw Error.invalid("invalid dense layer count") }
        mlpLayerTypes = text["mlp_layer_types"] as? [String]
            ?? (0..<layerCount).map { $0 < dense ? "dense" : "sparse" }
        layerTypes = (text["layer_types"] as? [String]
            ?? (0..<layerCount).map { $0 % 4 == 3 ? "deepseek_sparse_attention" : "linear_attention" })
            .map { $0 == "full_attention" ? "deepseek_sparse_attention" : $0 }
        guard mlpLayerTypes.count == layerCount, layerTypes.count == layerCount,
              mlpLayerTypes.allSatisfy({ ["dense", "sparse"].contains($0) }),
              layerTypes.allSatisfy({ ["linear_attention", "deepseek_sparse_attention"].contains($0) }) else {
            throw Error.invalid("invalid layer type schedule")
        }
        let linear = text["linear_attn_config"] as? [String: Any] ?? [:]
        linearHeads = linear["num_heads"] as? Int ?? (text["linear_num_heads"] as? Int ?? 64)
        linearHeadDim = linear["head_dim"] as? Int ?? (text["linear_head_dim"] as? Int ?? 128)
        linearConvKernel = linear["short_conv_kernel_size"] as? Int
            ?? (text["linear_conv_kernel_dim"] as? Int ?? 4)
        guard (1...1024).contains(linearHeads), (1...1024).contains(linearHeadDim),
              (1...1024).contains(linearConvKernel) else {
            throw Error.invalid("invalid linear attention dimensions")
        }
    }

    /// Single-sequence F32 recurrent matrices plus F32 short-convolution state.
    /// Excludes weights, sparse-attention caches, activations, and allocator overhead.
    public var recurrentStateBytes: Int {
        layerTypes.filter { $0 == "linear_attention" }.count * linearHeads
            * (linearHeadDim * linearHeadDim + 3 * linearHeadDim * (linearConvKernel - 1)) * 4
    }
}
