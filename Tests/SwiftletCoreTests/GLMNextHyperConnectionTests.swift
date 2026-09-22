import Foundation
import Testing
@testable import SwiftletCore

@Suite struct GLMNextHyperConnectionTests {
    private static let fixtureURL = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("fixtures/glm-hyper-connection/reference.json")

    /// FP32 op parity with the MLX CPU oracle. Sinkhorn's last column norm
    /// leaves about one epsilon of slack, so the invariant checks are wider.
    private static let parity: Float = 1e-6
    private static let balancedRowSum: Float = 1e-4

    private func document() throws -> [String: Any] {
        let data = try Data(contentsOf: Self.fixtureURL)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func integer(_ object: [String: Any], _ key: String) throws -> Int {
        if let value = object[key] as? Int { return value }
        return try #require((object[key] as? NSNumber)?.intValue)
    }

    private func scalar(_ object: [String: Any], _ key: String) throws -> Float {
        try #require((object[key] as? NSNumber)?.floatValue)
    }

    private func floats(_ object: [String: Any], _ key: String) throws -> [Float] {
        let raw = try #require(object[key] as? [Any])
        return try raw.map { try #require(($0 as? NSNumber)?.floatValue) }
    }

    private func config(_ object: [String: Any], iterations: Int? = nil) throws -> GLMNextHyperConnection.Config {
        try GLMNextHyperConnection.Config(
            hiddenSize: integer(object, "hidden_size"),
            hcMult: integer(object, "hc_mult"),
            hcEps: scalar(object, "hc_eps"),
            sinkhornIterations: iterations ?? integer(object, "sinkhorn_iters"),
            rmsNormEps: scalar(object, "rms_norm_eps")
        )
    }

    private func weights(_ object: [String: Any], _ config: GLMNextHyperConnection.Config) throws -> GLMNextHyperConnection.Weights {
        try GLMNextHyperConnection.Weights(
            fn: floats(object, "fn"), base: floats(object, "base"), scale: floats(object, "scale"), config: config
        )
    }

    private func maxAbs(_ actual: [Float], _ expected: [Float]) -> Float {
        guard actual.count == expected.count, !actual.isEmpty else { return .infinity }
        return zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
    }

    private func expectClose(_ actual: [Float], _ expected: [Float], _ tolerance: Float, _ label: String) {
        let error = maxAbs(actual, expected)
        #expect(actual.count == expected.count, "\(label) count \(actual.count) vs \(expected.count)")
        #expect(error <= tolerance, "\(label) max abs \(error)")
    }

    private func slice(_ values: [Float], batch: Int, tokens: Int, index: Int, stride: Int, alongTokens: Bool) -> [Float] {
        var output: [Float] = []
        if alongTokens {
            output.reserveCapacity(batch * stride)
            for item in 0..<batch {
                let start = (item * tokens + index) * stride
                output.append(contentsOf: values[start..<(start + stride)])
            }
        } else {
            let start = index * tokens * stride
            output.append(contentsOf: values[start..<(start + tokens * stride)])
        }
        return output
    }

    @Test func glmDefaultsMatchThePinnedTextConfig() throws {
        let config = try GLMNextHyperConnection.Config.glmDefault(hiddenSize: 4096)
        #expect(config.hcMult == 4)
        #expect(config.hcEps == 1e-6)
        #expect(config.sinkhornIterations == 20)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.mixCount == 24)
        #expect(config.featuresPerToken == 16_384)
        let root = try document()
        #expect(root["mlx"] as? String == "0.32.2")
        #expect(root["kernel_used"] as? Bool == false)
        #expect(root["extracted_ops"] as? [String] == ["_hc_split_sinkhorn_ops", "_hc_ops", "_hc_expand_op"])
    }

    @Test func rejectsInvalidDimensionsAndValues() throws {
        for attempt in [
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 0) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 16_385) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, hcMult: 0) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, hcMult: 33) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, sinkhornIterations: -1) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, sinkhornIterations: 129) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, hcEps: -1e-6) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, hcEps: .nan) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, hcEps: .infinity) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, rmsNormEps: 0) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, rmsNormEps: -1e-5) },
            { _ = try GLMNextHyperConnection.Config(hiddenSize: 8, rmsNormEps: .nan) },
        ] {
            #expect(throws: GLMNextHyperConnection.Error.self, performing: attempt)
        }
        let config = try GLMNextHyperConnection.Config.glmDefault(hiddenSize: 4)
        let mix = config.mixCount
        let width = config.featuresPerToken
        let fn = [Float](repeating: 0.1, count: mix * width)
        let base = [Float](repeating: 0.2, count: mix)
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.Weights(fn: Array(fn.dropLast()), base: base, scale: [1, 1, 1], config: config)
        }
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.Weights(fn: fn, base: Array(base.dropLast()), scale: [1, 1, 1], config: config)
        }
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.Weights(fn: fn, base: base, scale: [1, 1], config: config)
        }
        var nonFinite = fn
        nonFinite[3] = .nan
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.Weights(fn: nonFinite, base: base, scale: [1, 1, 1], config: config)
        }
        let weights = try GLMNextHyperConnection.Weights(fn: fn, base: base, scale: [0.5, 1.5, -0.25], config: config)
        let residual = [Float](repeating: 0.25, count: width)
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.collapse(residual: residual, batch: 0, tokens: 1, weights: weights, config: config)
        }
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.collapse(residual: residual, batch: 1, tokens: 0, weights: weights, config: config)
        }
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.collapse(residual: Array(residual.dropLast()), batch: 1, tokens: 1, weights: weights, config: config)
        }
        var badResidual = residual
        badResidual[0] = .infinity
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.coefficients(residual: badResidual, batch: 1, tokens: 1, weights: weights, config: config)
        }
        let narrow = try GLMNextHyperConnection.Config(hiddenSize: 4, hcMult: 2, sinkhornIterations: 1)
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.collapse(residual: residual, batch: 1, tokens: 1, weights: weights, config: narrow)
        }
        let collapsed = try GLMNextHyperConnection.collapse(residual: residual, batch: 1, tokens: 1, weights: weights, config: config)
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.expand(
                sublayer: [Float](repeating: 1, count: 3), residual: residual,
                post: collapsed.coefficients.post, comb: collapsed.coefficients.comb,
                batch: 1, tokens: 1, config: config
            )
        }
        var badComb = collapsed.coefficients.comb
        badComb[0] = .nan
        #expect(throws: GLMNextHyperConnection.Error.self) {
            _ = try GLMNextHyperConnection.expand(
                sublayer: [Float](repeating: 1, count: config.hiddenSize), residual: residual,
                post: collapsed.coefficients.post, comb: badComb, batch: 1, tokens: 1, config: config
            )
        }
    }

    @Test func fixturesMatchMLXOpsForEverySupportedWidth() throws {
        let cases = try #require(document()["cases"] as? [[String: Any]])
        #expect(cases.map { $0["name"] as? String } == ["glm_default", "hc2", "hc3", "hc1"])
        for spec in cases {
            let name = try #require(spec["name"] as? String)
            let built = try config(spec)
            let stored = try weights(spec, built)
            let batch = try integer(spec, "batch")
            let tokens = try integer(spec, "tokens")
            let residual = try floats(spec, "residual")
            let sublayer = try floats(spec, "sublayer")
            #expect(Set(stored.scale).count == 3, "\(name) scales must be nontrivial and distinct")
            #expect(stored.scale != [1, 1, 1], "\(name) scales must not be the default ones")
            let collapsed = try GLMNextHyperConnection.collapse(
                residual: residual, batch: batch, tokens: tokens, weights: stored, config: built
            )
            let again = try GLMNextHyperConnection.coefficients(
                residual: residual, batch: batch, tokens: tokens, weights: stored, config: built
            )
            #expect(collapsed.coefficients == again)
            expectClose(collapsed.coefficients.mixes, try floats(spec, "mixes"), Self.parity, "\(name) mixes")
            expectClose(collapsed.coefficients.pre, try floats(spec, "pre"), Self.parity, "\(name) pre")
            expectClose(collapsed.coefficients.post, try floats(spec, "post"), Self.parity, "\(name) post")
            expectClose(collapsed.coefficients.comb, try floats(spec, "comb"), Self.parity, "\(name) comb")
            expectClose(collapsed.collapsed, try floats(spec, "collapsed"), Self.parity, "\(name) collapsed")
            let branch = try GLMNextHyperConnection.applyBranch(
                residual: residual, sublayer: sublayer, batch: batch, tokens: tokens, weights: stored, config: built
            )
            expectClose(branch.streams, try floats(spec, "expanded"), Self.parity, "\(name) expanded")
            expectClose(branch.collapsed, collapsed.collapsed, 0, "\(name) branch collapse")
            if let zero = spec["zero_token"] as? [String: Any] {
                let token = try integer(zero, "token")
                let item = try integer(zero, "batch")
                let start = (item * tokens + token) * built.hiddenSize
                let hidden = branch.collapsed[start..<(start + built.hiddenSize)]
                #expect(hidden.allSatisfy { $0 == 0 }, "\(name) zero residual must collapse to zero")
            }
            if built.hcMult >= 3 {
                let transposed = transpose(collapsed.coefficients.comb, count: batch * tokens, streams: built.hcMult)
                #expect(maxAbs(transposed, collapsed.coefficients.comb) > Float(1e-3), "\(name) comb is symmetric")
                let wrong = try GLMNextHyperConnection.expand(
                    sublayer: sublayer, residual: residual, post: collapsed.coefficients.post, comb: transposed,
                    batch: batch, tokens: tokens, config: built
                )
                #expect(maxAbs(wrong, try floats(spec, "expanded")) > Float(1e-2), "\(name) transpose was not detectable")
            }
        }
    }

    @Test func sequentialTokensAndBatchesMatchTheBatchedOracle() throws {
        let spec = try #require((document()["cases"] as? [[String: Any]])?.first { $0["name"] as? String == "glm_default" })
        let built = try config(spec)
        let stored = try weights(spec, built)
        let batch = try integer(spec, "batch")
        let tokens = try integer(spec, "tokens")
        let residual = try floats(spec, "residual")
        let sublayer = try floats(spec, "sublayer")
        let expectedCollapsed = try floats(spec, "collapsed")
        let expectedExpanded = try floats(spec, "expanded")
        let expectedMixes = try floats(spec, "mixes")
        var collapsed: [Float] = []
        var expanded: [Float] = []
        var mixes: [Float] = []
        for token in 0..<tokens {
            let tokenResidual = slice(residual, batch: batch, tokens: tokens, index: token,
                                      stride: built.featuresPerToken, alongTokens: true)
            let tokenSublayer = slice(sublayer, batch: batch, tokens: tokens, index: token,
                                      stride: built.hiddenSize, alongTokens: true)
            let branch = try GLMNextHyperConnection.applyBranch(
                residual: tokenResidual, sublayer: tokenSublayer, batch: batch, tokens: 1,
                weights: stored, config: built
            )
            for item in 0..<batch {
                collapsed.append(contentsOf: branch.collapsed[item * built.hiddenSize..<(item + 1) * built.hiddenSize])
                expanded.append(contentsOf: branch.streams[item * built.featuresPerToken..<(item + 1) * built.featuresPerToken])
                mixes.append(contentsOf: branch.coefficients.mixes[item * built.mixCount..<(item + 1) * built.mixCount])
            }
        }
        // Token calls are ordered token-major here, while the oracle is batch-major.
        var oracleCollapsed: [Float] = []
        var oracleExpanded: [Float] = []
        var oracleMixes: [Float] = []
        for token in 0..<tokens {
            oracleCollapsed.append(contentsOf: slice(expectedCollapsed, batch: batch, tokens: tokens, index: token,
                                                     stride: built.hiddenSize, alongTokens: true))
            oracleExpanded.append(contentsOf: slice(expectedExpanded, batch: batch, tokens: tokens, index: token,
                                                    stride: built.featuresPerToken, alongTokens: true))
            oracleMixes.append(contentsOf: slice(expectedMixes, batch: batch, tokens: tokens, index: token,
                                                 stride: built.mixCount, alongTokens: true))
        }
        expectClose(collapsed, oracleCollapsed, Self.parity, "per-token collapsed")
        expectClose(expanded, oracleExpanded, Self.parity, "per-token expanded")
        expectClose(mixes, oracleMixes, Self.parity, "per-token mixes")

        var batchCollapsed: [Float] = []
        var batchExpanded: [Float] = []
        for item in 0..<batch {
            let itemResidual = slice(residual, batch: batch, tokens: tokens, index: item,
                                     stride: built.featuresPerToken, alongTokens: false)
            let itemSublayer = slice(sublayer, batch: batch, tokens: tokens, index: item,
                                     stride: built.hiddenSize, alongTokens: false)
            let branch = try GLMNextHyperConnection.applyBranch(
                residual: itemResidual, sublayer: itemSublayer, batch: 1, tokens: tokens,
                weights: stored, config: built
            )
            batchCollapsed.append(contentsOf: branch.collapsed)
            batchExpanded.append(contentsOf: branch.streams)
        }
        expectClose(batchCollapsed, expectedCollapsed, Self.parity, "per-batch collapsed")
        expectClose(batchExpanded, expectedExpanded, Self.parity, "per-batch expanded")
    }

    @Test func attentionThenFeedForwardUsesTheExpandedStream() throws {
        let spec = try #require(document()["sequential"] as? [String: Any])
        let built = try config(spec)
        let batch = try integer(spec, "batch")
        let tokens = try integer(spec, "tokens")
        let attention = try #require(spec["attention"] as? [String: Any])
        let feedForward = try #require(spec["feed_forward"] as? [String: Any])
        let attentionWeights = try weights(attention, built)
        let feedForwardWeights = try weights(feedForward, built)
        var residual = try floats(spec, "residual")
        let first = try GLMNextHyperConnection.applyBranch(
            residual: residual, sublayer: floats(attention, "sublayer"), batch: batch, tokens: tokens,
            weights: attentionWeights, config: built
        )
        expectClose(first.collapsed, try floats(attention, "collapsed"), Self.parity, "attention collapsed")
        expectClose(first.coefficients.comb, try floats(attention, "comb"), Self.parity, "attention comb")
        expectClose(first.streams, try floats(attention, "expanded"), Self.parity, "attention expanded")
        #expect(maxAbs(first.streams, residual) > Float(1e-2))
        residual = first.streams
        let second = try GLMNextHyperConnection.applyBranch(
            residual: residual, sublayer: floats(feedForward, "sublayer"), batch: batch, tokens: tokens,
            weights: feedForwardWeights, config: built
        )
        expectClose(second.coefficients.mixes, try floats(feedForward, "mixes"), Self.parity, "ffn mixes")
        expectClose(second.collapsed, try floats(feedForward, "collapsed"), Self.parity, "ffn collapsed")
        expectClose(second.coefficients.post, try floats(feedForward, "post"), Self.parity, "ffn post")
        expectClose(second.streams, try floats(feedForward, "expanded"), Self.parity, "ffn expanded")
        #expect(maxAbs(second.streams, first.streams) > Float(1e-2))

        var carried = try floats(spec, "residual")
        var chained = [Float](repeating: 0, count: carried.count)
        for stage in [attention, feedForward] {
            let stageWeights = try weights(stage, built)
            let sublayer = try floats(stage, "sublayer")
            var next = [Float]()
            next.reserveCapacity(carried.count)
            for token in 0..<tokens {
                let branch = try GLMNextHyperConnection.applyBranch(
                    residual: slice(carried, batch: batch, tokens: tokens, index: token,
                                    stride: built.featuresPerToken, alongTokens: true),
                    sublayer: slice(sublayer, batch: batch, tokens: tokens, index: token,
                                    stride: built.hiddenSize, alongTokens: true),
                    batch: batch, tokens: 1, weights: stageWeights, config: built
                )
                next.append(contentsOf: branch.streams)
            }
            carried = next
            chained = next
        }
        let oracle = try floats(feedForward, "expanded")
        var tokenMajor = [Float]()
        for token in 0..<tokens {
            tokenMajor.append(contentsOf: slice(oracle, batch: batch, tokens: tokens, index: token,
                                                stride: built.featuresPerToken, alongTokens: true))
        }
        expectClose(chained, tokenMajor, Self.parity, "token-carried chain")
    }

    @Test func sinkhornIterationCountChangesOnlyTheCombination() throws {
        let spec = try #require(document()["iteration_sweep"] as? [String: Any])
        let steps = try #require(spec["iterations"] as? [[String: Any]])
        #expect(steps.compactMap { $0["sinkhorn_iters"] as? Int } == [0, 1, 2, 20])
        let stored = try weights(spec, config(spec, iterations: 20))
        let residual = try floats(spec, "residual")
        let sublayer = try floats(spec, "sublayer")
        let batch = try integer(spec, "batch")
        let tokens = try integer(spec, "tokens")
        var outputs: [Int: GLMNextHyperConnection.BranchResult] = [:]
        for step in steps {
            let iterations = try integer(step, "sinkhorn_iters")
            let built = try config(spec, iterations: iterations)
            let branch = try GLMNextHyperConnection.applyBranch(
                residual: residual, sublayer: sublayer, batch: batch, tokens: tokens,
                weights: stored, config: built
            )
            expectClose(branch.coefficients.mixes, try floats(spec, "mixes"), Self.parity, "iters \(iterations) mixes")
            expectClose(branch.coefficients.pre, try floats(spec, "pre"), Self.parity, "iters \(iterations) pre")
            expectClose(branch.coefficients.post, try floats(spec, "post"), Self.parity, "iters \(iterations) post")
            expectClose(branch.collapsed, try floats(step, "collapsed"), Self.parity, "iters \(iterations) collapsed")
            expectClose(branch.coefficients.comb, try floats(step, "comb"), Self.parity, "iters \(iterations) comb")
            expectClose(branch.streams, try floats(step, "expanded"), Self.parity, "iters \(iterations) expanded")
            outputs[iterations] = branch
        }
        let zero = try #require(outputs[0])
        let one = try #require(outputs[1])
        let two = try #require(outputs[2])
        let full = try #require(outputs[20])
        #expect(zero.collapsed == one.collapsed)
        #expect(one.collapsed == two.collapsed)
        #expect(two.collapsed == full.collapsed)
        #expect(zero.coefficients.comb == one.coefficients.comb)
        #expect(zero.streams == one.streams)
        #expect(maxAbs(two.coefficients.comb, one.coefficients.comb) > Float(1e-3))
        #expect(maxAbs(full.coefficients.comb, two.coefficients.comb) > Float(1e-3))
        #expect(maxAbs(two.streams, one.streams) > Float(1e-4))
    }

    @Test func normalizationInvariantsHoldWithinEpsilon() throws {
        let root = try document()
        let cases = try #require(root["cases"] as? [[String: Any]])
        for spec in cases {
            let name = try #require(spec["name"] as? String)
            let built = try config(spec)
            let result = try GLMNextHyperConnection.collapse(
                residual: floats(spec, "residual"), batch: integer(spec, "batch"), tokens: integer(spec, "tokens"),
                weights: weights(spec, built), config: built
            )
            try expectNormalized(result.coefficients, config: built, label: name, rowsBalanced: built.sinkhornIterations >= 20)
        }
        let sweep = try #require(root["iteration_sweep"] as? [String: Any])
        let steps = try #require(sweep["iterations"] as? [[String: Any]])
        for step in steps {
            let iterations = try integer(step, "sinkhorn_iters")
            let built = try config(sweep, iterations: iterations)
            let result = try GLMNextHyperConnection.collapse(
                residual: floats(sweep, "residual"), batch: integer(sweep, "batch"), tokens: integer(sweep, "tokens"),
                weights: weights(sweep, built), config: built
            )
            let balanced = iterations >= 20
            try expectNormalized(result.coefficients, config: built, label: "sweep \(iterations)", rowsBalanced: balanced)
            let rowError = rowSumError(result.coefficients.comb, streams: built.hcMult)
            if iterations <= 1 {
                #expect(rowError > Float(0.1), "one column normalization must leave rows unbalanced, error \(rowError)")
            }
        }
    }

    private func expectNormalized(_ coefficients: GLMNextHyperConnection.Coefficients,
                                  config: GLMNextHyperConnection.Config, label: String, rowsBalanced: Bool) throws {
        let eps = config.hcEps
        #expect(coefficients.pre.allSatisfy { $0.isFinite && $0 >= eps && $0 <= Float(1) + eps + Float(1e-6) }, "\(label) pre bounds")
        #expect(coefficients.post.allSatisfy { $0.isFinite && $0 > 0 && $0 <= 2 }, "\(label) post bounds")
        #expect(coefficients.comb.allSatisfy { $0.isFinite && $0 > 0 }, "\(label) comb positive")
        let columnTolerance = max(Float(1e-5), eps * 5)
        #expect(columnSumError(coefficients.comb, streams: config.hcMult) <= columnTolerance, "\(label) columns")
        if rowsBalanced {
            #expect(rowSumError(coefficients.comb, streams: config.hcMult) <= Self.balancedRowSum, "\(label) rows")
        }
    }

    private func rowSumError(_ comb: [Float], streams: Int) -> Float {
        var worst: Float = 0
        let matrices = comb.count / (streams * streams)
        for index in 0..<matrices {
            let base = index * streams * streams
            for row in 0..<streams {
                var sum: Float = 0
                for column in 0..<streams { sum += comb[base + row * streams + column] }
                worst = max(worst, abs(sum - 1))
            }
        }
        return worst
    }

    private func columnSumError(_ comb: [Float], streams: Int) -> Float {
        var worst: Float = 0
        let matrices = comb.count / (streams * streams)
        for index in 0..<matrices {
            let base = index * streams * streams
            for column in 0..<streams {
                var sum: Float = 0
                for row in 0..<streams { sum += comb[base + row * streams + column] }
                worst = max(worst, abs(sum - 1))
            }
        }
        return worst
    }

    private func transpose(_ comb: [Float], count: Int, streams: Int) -> [Float] {
        var output = comb
        for index in 0..<count {
            let base = index * streams * streams
            for row in 0..<streams {
                for column in 0..<streams {
                    output[base + row * streams + column] = comb[base + column * streams + row]
                }
            }
        }
        return output
    }
}
