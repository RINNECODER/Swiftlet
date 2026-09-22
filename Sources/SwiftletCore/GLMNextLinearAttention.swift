import Foundation

/// CPU float32 reference for one GLM-5-Next linear-attention block.
///
/// The math follows mlx-vlm `Glm5NextLinearAttention` plus `gated_delta_ops`
/// with the safe decay
/// `exp(lowerBound * sigmoid(exp(A_log) * (a + dtBias)))`.
/// `step` consumes one token and updates the causal short-convolution window
/// and the recurrent state. Calling it again continues the sequence.
/// Hyper-connections, sparse attention, padding masks, and the Metal kernel
/// are outside this type.
public struct GLMNextLinearAttention: Sendable {
    public enum Error: Swift.Error, Sendable, Equatable, CustomStringConvertible {
        case invalid(String)
        public var description: String {
            switch self {
            case .invalid(let message): return "GLM linear attention: \(message)"
            }
        }
    }

    /// Shapes for one block. Construction only checks integers; it does not
    /// allocate activations or production weights.
    public struct Configuration: Sendable, Equatable {
        public let hiddenSize: Int
        public let numHeads: Int
        public let headDim: Int
        public let convKernelSize: Int
        public let lowerBound: Float
        public let rmsNormEps: Float
        public let projectionDim: Int
        public let qkvChannels: Int
        public let convolutionStateCount: Int
        public let recurrentStateCount: Int
        public let qkvProjectionCount: Int
        public let qkvConvolutionCount: Int
        public let fbgAProjectionCount: Int
        public let fBProjectionCount: Int
        public let gBProjectionCount: Int
        public let dtBiasCount: Int
        public let outputProjectionCount: Int

        /// `1e-6 / headDim`, the weightless query/key RMSNorm epsilon in the module.
        public var queryKeyNormEps: Float { Float(1e-6 / Double(headDim)) }

        /// GLM-5.3-Flash text linear attention: hidden 4096, 64 heads of 128,
        /// conv kernel 4, safe-gate lower bound -5, RMSNorm epsilon 1e-5.
        public static func glm53FlashText() throws -> Configuration {
            try Configuration(
                hiddenSize: 4096,
                numHeads: 64,
                headDim: 128,
                convKernelSize: 4,
                lowerBound: -5,
                rmsNormEps: 1e-5
            )
        }

        public init(hiddenSize: Int, numHeads: Int, headDim: Int, convKernelSize: Int,
                    lowerBound: Float, rmsNormEps: Float) throws {
            guard hiddenSize > 0, numHeads > 0, headDim > 0, convKernelSize > 0 else {
                throw Error.invalid("hidden size, heads, head dim, and conv kernel must be positive")
            }
            guard lowerBound.isFinite else {
                throw Error.invalid("safe-gate lower bound must be finite; the non-safe decay is not implemented")
            }
            guard rmsNormEps.isFinite, rmsNormEps > 0 else {
                throw Error.invalid("RMSNorm epsilon must be finite and positive")
            }
            self.hiddenSize = hiddenSize
            self.numHeads = numHeads
            self.headDim = headDim
            self.convKernelSize = convKernelSize
            self.lowerBound = lowerBound
            self.rmsNormEps = rmsNormEps
            projectionDim = try Self.product(numHeads, headDim, label: "projection")
            qkvChannels = try Self.product(3, projectionDim, label: "qkv channels")
            let history = convKernelSize - 1
            convolutionStateCount = history == 0
                ? 0
                : try Self.product(history, qkvChannels, label: "convolution state")
            recurrentStateCount = try Self.product(numHeads, headDim, headDim, label: "recurrent state")
            qkvProjectionCount = try Self.product(qkvChannels, hiddenSize, label: "qkv projection")
            qkvConvolutionCount = try Self.product(qkvChannels, convKernelSize, label: "qkv convolution")
            let fbgRows = try Self.sum(
                try Self.product(2, headDim, label: "gate features"),
                numHeads,
                label: "fbg rows"
            )
            fbgAProjectionCount = try Self.product(fbgRows, hiddenSize, label: "fbg projection")
            fBProjectionCount = try Self.product(projectionDim, headDim, label: "f_b projection")
            gBProjectionCount = fBProjectionCount
            dtBiasCount = projectionDim
            outputProjectionCount = try Self.product(hiddenSize, projectionDim, label: "output projection")
        }

        private static func product(_ factors: Int..., label: String) throws -> Int {
            var result = 1
            for factor in factors {
                let (next, overflow) = result.multipliedReportingOverflow(by: factor)
                guard !overflow, next > 0 else {
                    throw Error.invalid("\(label) dimensions overflow")
                }
                result = next
            }
            return result
        }

        private static func sum(_ lhs: Int, _ rhs: Int, label: String) throws -> Int {
            let (next, overflow) = lhs.addingReportingOverflow(rhs)
            guard !overflow, next > 0 else {
                throw Error.invalid("\(label) dimensions overflow")
            }
            return next
        }
    }

    /// Explicit float weights in MLX row-major layout, ready for a checkpoint adapter.
    ///
    /// - `qkvProjection`: `qkv_proj.weight`, `(3 * heads * headDim, hidden)`
    /// - `qkvConvolution`: `qkv_conv.conv.weight`, `(channels, kernel, 1)`
    /// - `fbgAProjection`: `fbg_a_proj.weight`, `(2 * headDim + heads, hidden)`
    /// - `fBProjection`: `f_b_proj.weight`, `(heads * headDim, headDim)`
    /// - `gBProjection`: `g_b_proj.weight`, `(heads * headDim, headDim)`
    /// - `aLog`: `A_log`, `(heads,)`
    /// - `dtBias`: `dt_bias`, `(heads, headDim)`
    /// - `outputNorm`: `o_norm.weight`, `(headDim,)`
    /// - `outputProjection`: `o_proj.weight`, `(hidden, heads * headDim)`
    public struct Weights: Sendable {
        public let qkvProjection: [Float]
        public let qkvConvolution: [Float]
        public let fbgAProjection: [Float]
        public let fBProjection: [Float]
        public let gBProjection: [Float]
        public let aLog: [Float]
        public let dtBias: [Float]
        public let outputNorm: [Float]
        public let outputProjection: [Float]

        public init(configuration: Configuration, qkvProjection: [Float], qkvConvolution: [Float],
                    fbgAProjection: [Float], fBProjection: [Float], gBProjection: [Float],
                    aLog: [Float], dtBias: [Float], outputNorm: [Float],
                    outputProjection: [Float]) throws {
            try Self.require(qkvProjection, configuration.qkvProjectionCount, "qkv projection")
            try Self.require(qkvConvolution, configuration.qkvConvolutionCount, "qkv convolution")
            try Self.require(fbgAProjection, configuration.fbgAProjectionCount, "fbg projection")
            try Self.require(fBProjection, configuration.fBProjectionCount, "f_b projection")
            try Self.require(gBProjection, configuration.gBProjectionCount, "g_b projection")
            try Self.require(aLog, configuration.numHeads, "A_log")
            try Self.require(dtBias, configuration.dtBiasCount, "dt_bias")
            try Self.require(outputNorm, configuration.headDim, "output norm")
            try Self.require(outputProjection, configuration.outputProjectionCount, "output projection")
            self.qkvProjection = qkvProjection
            self.qkvConvolution = qkvConvolution
            self.fbgAProjection = fbgAProjection
            self.fBProjection = fBProjection
            self.gBProjection = gBProjection
            self.aLog = aLog
            self.dtBias = dtBias
            self.outputNorm = outputNorm
            self.outputProjection = outputProjection
        }

        private static func require(_ values: [Float], _ count: Int, _ name: String) throws {
            guard values.count == count else {
                throw Error.invalid("\(name) has \(values.count) values, expected \(count)")
            }
            guard values.allSatisfy(\.isFinite) else {
                throw Error.invalid("\(name) must be finite")
            }
        }
    }

    /// Single-sequence state. Convolution is time-major `(kernel - 1, channels)`
    /// with the oldest row first, and empty when the kernel is 1. Recurrent
    /// state is `(heads, headDim, headDim)`, value row then key column.
    public struct State: Sendable, Equatable {
        public var convolution: [Float]
        public var recurrent: [Float]

        public init(configuration: Configuration) {
            convolution = [Float](repeating: 0, count: configuration.convolutionStateCount)
            recurrent = [Float](repeating: 0, count: configuration.recurrentStateCount)
        }

        public mutating func reset() {
            convolution = [Float](repeating: 0, count: convolution.count)
            recurrent = [Float](repeating: 0, count: recurrent.count)
        }
    }

    public let configuration: Configuration
    public let weights: Weights

    public init(configuration: Configuration, weights: Weights) throws {
        guard weights.qkvProjection.count == configuration.qkvProjectionCount,
              weights.qkvConvolution.count == configuration.qkvConvolutionCount,
              weights.fbgAProjection.count == configuration.fbgAProjectionCount,
              weights.fBProjection.count == configuration.fBProjectionCount,
              weights.gBProjection.count == configuration.gBProjectionCount,
              weights.aLog.count == configuration.numHeads,
              weights.dtBias.count == configuration.dtBiasCount,
              weights.outputNorm.count == configuration.headDim,
              weights.outputProjection.count == configuration.outputProjectionCount else {
            throw Error.invalid("weights do not match the configuration")
        }
        self.configuration = configuration
        self.weights = weights
    }

    public func makeState() -> State {
        State(configuration: configuration)
    }

    /// One token. `token` is a hidden vector. The same `state` carries the
    /// convolution window and recurrent matrix into the next call.
    public func step(_ token: [Float], state: inout State) throws -> [Float] {
        let config = configuration
        guard token.count == config.hiddenSize else {
            throw Error.invalid("token length \(token.count) != hidden size \(config.hiddenSize)")
        }
        guard token.allSatisfy(\.isFinite) else {
            throw Error.invalid("token must be finite")
        }
        guard state.convolution.count == config.convolutionStateCount,
              state.recurrent.count == config.recurrentStateCount else {
            throw Error.invalid("state dimensions do not match the configuration")
        }

        let qkv = project(
            weights.qkvProjection, token,
            rows: config.qkvChannels, columns: config.hiddenSize
        )
        let conv = convolvedQKV(qkv, state: &state)
        let (q, k, v) = normalizedQueriesAndKeys(conv)
        let features = project(
            weights.fbgAProjection, token,
            rows: config.fbgAProjectionCount / config.hiddenSize, columns: config.hiddenSize
        )
        let headDim = config.headDim
        let heads = config.numHeads
        let fA = Array(features[0..<headDim])
        let beta = features[headDim..<(headDim + heads)].map { sigmoid($0) }
        let gA = Array(features[(headDim + heads)..<(2 * headDim + heads)])
        let a = project(weights.fBProjection, fA, rows: config.projectionDim, columns: headDim)
        let gateLogits = project(weights.gBProjection, gA, rows: config.projectionDim, columns: headDim)
        let mixed = recur(q, k, v, a, beta, state: &state)
        let gated = applyOutputGate(mixed, gateLogits)
        return project(
            weights.outputProjection, gated,
            rows: config.hiddenSize, columns: config.projectionDim
        )
    }

    private func convolvedQKV(_ qkv: [Float], state: inout State) -> [Float] {
        let channels = configuration.qkvChannels
        let kernel = configuration.convKernelSize
        let history = kernel - 1
        var output = [Float](repeating: 0, count: channels)
        for channel in 0..<channels {
            var accumulated: Float = 0
            for lag in 0..<history {
                let weight = weights.qkvConvolution[channel * kernel + lag]
                accumulated += weight * state.convolution[lag * channels + channel]
            }
            accumulated += weights.qkvConvolution[channel * kernel + history] * qkv[channel]
            output[channel] = silu(accumulated)
        }
        if history > 0 {
            var next = [Float]()
            next.reserveCapacity(history * channels)
            if history > 1 {
                next.append(contentsOf: state.convolution[channels..<(history * channels)])
            }
            next.append(contentsOf: qkv)
            state.convolution = next
        }
        return output
    }

    private func normalizedQueriesAndKeys(_ conv: [Float]) -> (q: [Float], k: [Float], v: [Float]) {
        let width = configuration.projectionDim
        let dim = configuration.headDim
        var q = Array(conv[0..<width])
        var k = Array(conv[width..<(2 * width)])
        let v = Array(conv[(2 * width)..<(3 * width)])
        // Weightless RMSNorm over the head. q uses 1/headDim; k uses 1/sqrt(headDim).
        rmsNorm(&q, rows: configuration.numHeads, dim: dim, weight: nil, eps: configuration.queryKeyNormEps)
        rmsNorm(&k, rows: configuration.numHeads, dim: dim, weight: nil, eps: configuration.queryKeyNormEps)
        let scale = 1 / Double(dim).squareRoot()
        let qScale = Float(scale * scale)
        let kScale = Float(scale)
        for index in q.indices {
            q[index] *= qScale
            k[index] *= kScale
        }
        return (q, k, v)
    }

    private func recur(_ q: [Float], _ k: [Float], _ v: [Float], _ a: [Float], _ beta: [Float],
                       state: inout State) -> [Float] {
        let heads = configuration.numHeads
        let dim = configuration.headDim
        let lowerBound = configuration.lowerBound
        var output = [Float](repeating: 0, count: configuration.projectionDim)
        for head in 0..<heads {
            let expA = exp(weights.aLog[head])
            let offset = head * dim
            var decay = [Float](repeating: 0, count: dim)
            for dk in 0..<dim {
                let mixed = expA * (a[offset + dk] + weights.dtBias[offset + dk])
                decay[dk] = exp(lowerBound * sigmoid(mixed))
            }
            for dv in 0..<dim {
                let stateBase = (head * dim + dv) * dim
                var kv: Float = 0
                for dk in 0..<dim {
                    state.recurrent[stateBase + dk] *= decay[dk]
                    kv += state.recurrent[stateBase + dk] * k[offset + dk]
                }
                let delta = (v[offset + dv] - kv) * beta[head]
                var projected: Float = 0
                for dk in 0..<dim {
                    state.recurrent[stateBase + dk] += k[offset + dk] * delta
                    projected += state.recurrent[stateBase + dk] * q[offset + dk]
                }
                output[offset + dv] = projected
            }
        }
        return output
    }

    private func applyOutputGate(_ y: [Float], _ gateLogits: [Float]) -> [Float] {
        var normed = y
        rmsNorm(
            &normed, rows: configuration.numHeads, dim: configuration.headDim,
            weight: weights.outputNorm, eps: configuration.rmsNormEps
        )
        for index in normed.indices {
            normed[index] *= sigmoid(gateLogits[index])
        }
        return normed
    }

    private func project(_ matrix: [Float], _ vector: [Float], rows: Int, columns: Int) -> [Float] {
        var output = [Float](repeating: 0, count: rows)
        var row = 0
        for index in 0..<rows {
            var accumulated: Float = 0
            let base = row
            for column in 0..<columns {
                accumulated += matrix[base + column] * vector[column]
            }
            output[index] = accumulated
            row += columns
        }
        return output
    }

    private func rmsNorm(_ values: inout [Float], rows: Int, dim: Int, weight: [Float]?, eps: Float) {
        for row in 0..<rows {
            let base = row * dim
            var sum: Float = 0
            for index in 0..<dim {
                let value = values[base + index]
                sum += value * value
            }
            let scale = 1 / (sum / Float(dim) + eps).squareRoot()
            if let weight {
                for index in 0..<dim {
                    values[base + index] *= scale * weight[index]
                }
            } else {
                for index in 0..<dim {
                    values[base + index] *= scale
                }
            }
        }
    }

    private func sigmoid(_ value: Float) -> Float {
        if value >= 0 {
            let z = exp(-value)
            return 1 / (1 + z)
        }
        let z = exp(value)
        return z / (1 + z)
    }

    private func silu(_ value: Float) -> Float {
        value * sigmoid(value)
    }
}
