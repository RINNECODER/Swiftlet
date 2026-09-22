import Foundation

/// CPU reference for GLM manifold-constrained hyper-connections.
///
/// The math follows the FP32 ops in mlx-vlm `hyper_connection.py`
/// (`_hc_split_sinkhorn_ops`, `_hc_ops`, `_hc_expand_op`), not the fused
/// Metal kernel. One value surrounds attention and another surrounds the MLP.
/// This type does not run either sublayer: the caller RMS-normalizes
/// `collapsed`, runs the sublayer, and passes that output to `expand`.
///
/// Layout is row-major. `residual` is `(batch, tokens, hcMult, hiddenSize)`.
/// `fn` is `(mixCount, hcMult * hiddenSize)` and mixes with `z @ fn.T`.
/// `base` is `(mixCount,)` split into pre, post, then a row-major combination.
/// `scale` is `(pre, post, combination)`. The combination matrix is row-major
/// `(hcMult, hcMult)`.
public enum GLMNextHyperConnection {
    public enum Error: Swift.Error, Equatable, CustomStringConvertible {
        case invalid(String)
        public var description: String {
            switch self { case .invalid(let message): return "GLM hyper-connection: \(message)" }
        }
    }

    /// Defaults match the pinned GLM text config: `hc_mult` 4, `hc_eps` 1e-6,
    /// 20 Sinkhorn iterations, and RMS epsilon 1e-5.
    public struct Config: Sendable, Equatable {
        public static let glmHCMult = 4
        public static let glmHCEps: Float = 1e-6
        public static let glmSinkhornIterations = 20
        public static let glmRMSNormEps: Float = 1e-5

        public let hiddenSize: Int
        public let hcMult: Int
        public let hcEps: Float
        public let sinkhornIterations: Int
        public let rmsNormEps: Float

        public var mixCount: Int { (2 + hcMult) * hcMult }
        public var featuresPerToken: Int { hcMult * hiddenSize }

        public init(hiddenSize: Int, hcMult: Int = glmHCMult, hcEps: Float = glmHCEps,
                    sinkhornIterations: Int = glmSinkhornIterations, rmsNormEps: Float = glmRMSNormEps) throws {
            guard (1...16_384).contains(hiddenSize) else { throw Error.invalid("invalid hidden size") }
            guard (1...32).contains(hcMult) else { throw Error.invalid("invalid hc_mult") }
            guard (0...128).contains(sinkhornIterations) else { throw Error.invalid("invalid Sinkhorn iterations") }
            guard hcEps.isFinite, (0...1).contains(hcEps) else { throw Error.invalid("invalid hc_eps") }
            guard rmsNormEps.isFinite, rmsNormEps > 0, rmsNormEps <= 1 else {
                throw Error.invalid("invalid rms_norm_eps")
            }
            self.hiddenSize = hiddenSize
            self.hcMult = hcMult
            self.hcEps = hcEps
            self.sinkhornIterations = sinkhornIterations
            self.rmsNormEps = rmsNormEps
        }

        public static func glmDefault(hiddenSize: Int) throws -> Config {
            try Config(hiddenSize: hiddenSize)
        }
    }

    public struct Weights: Sendable, Equatable {
        public let fn: [Float]
        public let base: [Float]
        public let scale: [Float]

        public init(fn: [Float], base: [Float], scale: [Float], config: Config) throws {
            guard fn.count == config.mixCount * config.featuresPerToken,
                  base.count == config.mixCount, scale.count == 3,
                  fn.allSatisfy(\.isFinite), base.allSatisfy(\.isFinite), scale.allSatisfy(\.isFinite) else {
                throw Error.invalid("invalid hyper-connection weights")
            }
            self.fn = fn
            self.base = base
            self.scale = scale
        }
    }

    /// FP32 mix projection and the pre, post, and combination gates derived from it.
    public struct Coefficients: Sendable, Equatable {
        public let mixes: [Float]
        public let pre: [Float]
        public let post: [Float]
        public let comb: [Float]
    }

    public struct CollapseResult: Sendable, Equatable {
        public let collapsed: [Float]
        public let coefficients: Coefficients
    }

    public struct BranchResult: Sendable, Equatable {
        public let collapsed: [Float]
        public let coefficients: Coefficients
        /// Expanded streams, `(batch, tokens, hcMult, hiddenSize)`.
        public let streams: [Float]
    }

    public static func coefficients(residual: [Float], batch: Int, tokens: Int,
                                    weights: Weights, config: Config) throws -> Coefficients {
        try evaluate(residual: residual, batch: batch, tokens: tokens, weights: weights, config: config).coefficients
    }

    public static func collapse(residual: [Float], batch: Int, tokens: Int,
                                weights: Weights, config: Config) throws -> CollapseResult {
        try evaluate(residual: residual, batch: batch, tokens: tokens, weights: weights, config: config)
    }

    /// `output[h, d] = post[h] * sublayer[d] + sum_k comb[k, h] * residual[k, d]`.
    /// The combination is transposed on its last two axes before the matmul,
    /// and the residual is the original stream, not the collapsed vector.
    public static func expand(sublayer: [Float], residual: [Float], post: [Float], comb: [Float],
                              batch: Int, tokens: Int, config: Config) throws -> [Float] {
        let count = try tokenCount(batch, tokens, config)
        let hidden = config.hiddenSize
        let streams = config.hcMult
        let width = config.featuresPerToken
        guard residual.count == count * width, sublayer.count == count * hidden,
              post.count == count * streams, comb.count == count * streams * streams,
              residual.allSatisfy(\.isFinite), sublayer.allSatisfy(\.isFinite),
              post.allSatisfy(\.isFinite), comb.allSatisfy(\.isFinite) else {
            throw Error.invalid("invalid expand inputs")
        }
        var output = [Float](repeating: 0, count: count * width)
        for index in 0..<count {
            let residualBase = index * width
            let sublayerBase = index * hidden
            let postBase = index * streams
            let combBase = index * streams * streams
            let outputBase = index * width
            for stream in 0..<streams {
                for dim in 0..<hidden {
                    var mixed: Float = post[postBase + stream] * sublayer[sublayerBase + dim]
                    for source in 0..<streams {
                        mixed += comb[combBase + source * streams + stream] * residual[residualBase + source * hidden + dim]
                    }
                    output[outputBase + stream * hidden + dim] = mixed
                }
            }
        }
        return output
    }

    public static func applyBranch(residual: [Float], sublayer: [Float], batch: Int, tokens: Int,
                                   weights: Weights, config: Config) throws -> BranchResult {
        let collapsed = try collapse(residual: residual, batch: batch, tokens: tokens, weights: weights, config: config)
        let streams = try expand(sublayer: sublayer, residual: residual, post: collapsed.coefficients.post,
                                 comb: collapsed.coefficients.comb, batch: batch, tokens: tokens, config: config)
        return BranchResult(collapsed: collapsed.collapsed, coefficients: collapsed.coefficients, streams: streams)
    }

    private static func tokenCount(_ batch: Int, _ tokens: Int, _ config: Config) throws -> Int {
        guard batch > 0, tokens > 0, batch <= Int.max / tokens else {
            throw Error.invalid("invalid batch or token count")
        }
        let count = batch * tokens
        guard count <= Int.max / config.featuresPerToken else {
            throw Error.invalid("invalid batch or token count")
        }
        return count
    }

    private static func evaluate(residual: [Float], batch: Int, tokens: Int,
                                 weights: Weights, config: Config) throws -> CollapseResult {
        guard weights.fn.count == config.mixCount * config.featuresPerToken,
              weights.base.count == config.mixCount, weights.scale.count == 3 else {
            throw Error.invalid("weights do not match the hyper-connection config")
        }
        let count = try tokenCount(batch, tokens, config)
        let hidden = config.hiddenSize
        let streams = config.hcMult
        let width = config.featuresPerToken
        let mixCount = config.mixCount
        guard residual.count == count * width, residual.allSatisfy(\.isFinite) else {
            throw Error.invalid("invalid residual")
        }
        var mixes = [Float](repeating: 0, count: count * mixCount)
        var pre = [Float](repeating: 0, count: count * streams)
        var post = [Float](repeating: 0, count: count * streams)
        var comb = [Float](repeating: 0, count: count * streams * streams)
        var collapsed = [Float](repeating: 0, count: count * hidden)
        let preScale = weights.scale[0]
        let postScale = weights.scale[1]
        let combScale = weights.scale[2]
        for index in 0..<count {
            let residualBase = index * width
            // Weightless RMS over the flattened streams. Collapse still reads
            // the original residual below; only the mix logits use this norm.
            var sumSquares: Float = 0
            for offset in 0..<width {
                let value = residual[residualBase + offset]
                sumSquares += value * value
            }
            let inverseRMS = 1 / sqrt(sumSquares / Float(width) + config.rmsNormEps)
            let mixBase = index * mixCount
            for row in 0..<mixCount {
                let fnBase = row * width
                var dot: Float = 0
                for offset in 0..<width {
                    dot += residual[residualBase + offset] * inverseRMS * weights.fn[fnBase + offset]
                }
                mixes[mixBase + row] = dot
            }
            let gateBase = index * streams
            for stream in 0..<streams {
                pre[gateBase + stream] = sigmoid(mixes[mixBase + stream] * preScale + weights.base[stream]) + config.hcEps
                post[gateBase + stream] = 2 * sigmoid(
                    mixes[mixBase + streams + stream] * postScale + weights.base[streams + stream]
                )
            }
            let combBase = index * streams * streams
            let logitBase = mixBase + 2 * streams
            let baseOffset = 2 * streams
            for row in 0..<streams {
                for column in 0..<streams {
                    let flat = row * streams + column
                    comb[combBase + flat] = mixes[logitBase + flat] * combScale + weights.base[baseOffset + flat]
                }
            }
            sinkhorn(&comb, base: combBase, streams: streams, iterations: config.sinkhornIterations, eps: config.hcEps)
            let collapsedBase = index * hidden
            for dim in 0..<hidden {
                var sum: Float = 0
                for stream in 0..<streams {
                    sum += pre[gateBase + stream] * residual[residualBase + stream * hidden + dim]
                }
                collapsed[collapsedBase + dim] = sum
            }
        }
        return CollapseResult(
            collapsed: collapsed,
            coefficients: Coefficients(mixes: mixes, pre: pre, post: post, comb: comb)
        )
    }

    /// Initial column normalization always runs. Further iterations are
    /// `max(sinkhornIterations - 1, 0)` row-then-column passes, so 0 and 1 match.
    private static func sinkhorn(_ comb: inout [Float], base: Int, streams: Int, iterations: Int, eps: Float) {
        for row in 0..<streams {
            let rowBase = base + row * streams
            var maxLogit = comb[rowBase]
            if streams > 1 {
                for column in 1..<streams { maxLogit = max(maxLogit, comb[rowBase + column]) }
            }
            var sum: Float = 0
            for column in 0..<streams {
                let value = exp(comb[rowBase + column] - maxLogit)
                comb[rowBase + column] = value
                sum += value
            }
            for column in 0..<streams {
                comb[rowBase + column] = comb[rowBase + column] / sum + eps
            }
        }
        normalizeColumns(&comb, base: base, streams: streams, eps: eps)
        for _ in 0..<max(iterations - 1, 0) {
            normalizeRows(&comb, base: base, streams: streams, eps: eps)
            normalizeColumns(&comb, base: base, streams: streams, eps: eps)
        }
    }

    private static func normalizeRows(_ comb: inout [Float], base: Int, streams: Int, eps: Float) {
        for row in 0..<streams {
            let rowBase = base + row * streams
            var sum: Float = 0
            for column in 0..<streams { sum += comb[rowBase + column] }
            for column in 0..<streams { comb[rowBase + column] /= sum + eps }
        }
    }

    private static func normalizeColumns(_ comb: inout [Float], base: Int, streams: Int, eps: Float) {
        var columnSums = [Float](repeating: 0, count: streams)
        for row in 0..<streams {
            let rowBase = base + row * streams
            for column in 0..<streams { columnSums[column] += comb[rowBase + column] }
        }
        for row in 0..<streams {
            let rowBase = base + row * streams
            for column in 0..<streams { comb[rowBase + column] /= columnSums[column] + eps }
        }
    }

    private static func sigmoid(_ value: Float) -> Float {
        if value >= 0 {
            let z = exp(-value)
            return 1 / (1 + z)
        }
        let z = exp(value)
        return z / (1 + z)
    }
}
