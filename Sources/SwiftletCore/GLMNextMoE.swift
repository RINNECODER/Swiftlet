import Foundation

/// CPU correctness reference for one GLM feed-forward block. Routed experts
/// are read one at a time from storage; no full model or expert stack is loaded.
/// This is a port-validation primitive, not an optimized inference backend.
public enum GLMNextMoE {
    public struct Route: Sendable {
        public let expert: Int
        public let weight: Float
    }

    public static func route(logits: [Float], correctionBias: [Float], config: GLMNextConfig) throws -> [Route] {
        guard logits.count == config.expertCount, correctionBias.count == logits.count,
              logits.allSatisfy(\.isFinite), correctionBias.allSatisfy(\.isFinite) else {
            throw GLMNextConfig.Error.invalid("invalid router logits/bias")
        }
        let scores = logits.map { 1 / (1 + exp(-$0)) }
        let choices = zip(scores, correctionBias).map(+)
        let perGroup = config.expertCount / config.groupCount
        var candidates = Array(logits.indices)
        if config.groupCount > 1 {
            let groupScores = (0..<config.groupCount).map { group -> Float in
                let start = group * perGroup
                return choices[start..<(start + perGroup)].sorted(by: >).prefix(2).reduce(0, +)
            }
            let kept = Set(groupScores.indices.sorted {
                groupScores[$0] == groupScores[$1] ? $0 < $1 : groupScores[$0] > groupScores[$1]
            }.prefix(config.topKGroups))
            candidates = candidates.filter { kept.contains($0 / perGroup) }
        }
        // Stable ties for the CPU oracle; MLX argpartition does not promise a
        // tie order. Correction bias changes selection, never mixture weights.
        let selected = candidates.sorted {
            choices[$0] == choices[$1] ? $0 < $1 : choices[$0] > choices[$1]
        }.prefix(config.topK)
        let denominator: Float = config.topK > 1 && config.normalizeTopK
            ? selected.reduce(Float(0)) { $0 + scores[$1] } + 1e-20 : 1
        return selected.map { Route(expert: $0, weight: scores[$0] / denominator * config.routedScalingFactor) }
    }

    private static func matvec(_ matrix: [Float], _ x: [Float], rows: Int) -> [Float] {
        (0..<rows).map { row in
            var value: Float = 0
            for column in x.indices { value += matrix[row * x.count + column] * x[column] }
            return value
        }
    }

    private static func activate(gate: [Float], up: [Float], limit: Float) -> [Float] {
        zip(gate, up).map { gate, up in
            let g = min(gate, limit)
            return (g / (1 + exp(-g))) * max(-limit, min(up, limit))
        }
    }

    private static func dense(_ checkpoint: GLMNextCheckpoint, prefix: String,
                              input: [Float], intermediate: Int) throws -> [Float] {
        let gate: [Float]
        let up: [Float]
        if checkpoint.contains(prefix + ".gate_up_proj.weight") {
            let joined = matvec(try checkpoint.matrix(prefix + ".gate_up_proj", rows: intermediate * 2,
                                                       columns: input.count), input, rows: intermediate * 2)
            gate = Array(joined.prefix(intermediate))
            up = Array(joined.suffix(intermediate))
        } else {
            gate = matvec(try checkpoint.matrix(prefix + ".gate_proj", rows: intermediate, columns: input.count),
                          input, rows: intermediate)
            up = matvec(try checkpoint.matrix(prefix + ".up_proj", rows: intermediate, columns: input.count),
                        input, rows: intermediate)
        }
        let hidden = activate(gate: gate, up: up, limit: checkpoint.config.swigluLimit)
        return matvec(try checkpoint.matrix(prefix + ".down_proj", rows: input.count, columns: intermediate),
                      hidden, rows: input.count)
    }

    /// Feed-forward result only: excludes the block's hyper-connections,
    /// normalization, residual mixing and attention.
    public static func forward(checkpoint: GLMNextCheckpoint, layer: Int, input: [Float]) throws -> [Float] {
        let config = checkpoint.config
        guard (0..<config.layerCount).contains(layer), input.count == config.hiddenSize,
              input.allSatisfy(\.isFinite) else { throw GLMNextConfig.Error.invalid("invalid feed-forward input") }
        let prefix = "model.layers.\(layer).mlp"
        if config.mlpLayerTypes[layer] == "dense" {
            return try dense(checkpoint, prefix: prefix, input: input, intermediate: config.intermediateSize)
        }
        let logits = matvec(try checkpoint.matrix(prefix + ".gate", rows: config.expertCount, columns: config.hiddenSize),
                            input, rows: config.expertCount)
        let bias = try checkpoint.vector(prefix + ".gate.e_score_correction_bias", count: config.expertCount)
        let routes = try route(logits: logits, correctionBias: bias, config: config)
        var output = [Float](repeating: 0, count: config.hiddenSize)
        for route in routes {
            let gate = matvec(try checkpoint.expertProjection(layer: layer, expert: route.expert, projection: "gate_proj"),
                              input, rows: config.moeIntermediateSize)
            let up = matvec(try checkpoint.expertProjection(layer: layer, expert: route.expert, projection: "up_proj"),
                            input, rows: config.moeIntermediateSize)
            let hidden = activate(gate: gate, up: up, limit: config.swigluLimit)
            let expert = matvec(try checkpoint.expertProjection(layer: layer, expert: route.expert, projection: "down_proj"),
                                hidden, rows: config.hiddenSize)
            for i in output.indices { output[i] += route.weight * expert[i] }
        }
        let shared = try dense(checkpoint, prefix: prefix + ".shared_experts", input: input,
                               intermediate: config.moeIntermediateSize * config.sharedExpertCount)
        // GLM's shared expert is added directly, without Qwen's sigmoid gate.
        for i in output.indices { output[i] += shared[i] }
        return output
    }
}
