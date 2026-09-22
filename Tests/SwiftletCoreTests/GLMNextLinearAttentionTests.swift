import Foundation
import Testing
@testable import SwiftletCore

@Suite struct GLMNextLinearAttentionTests {
    private struct Fixture: Decodable {
        struct Step: Decodable {
            let output: [Float]
            let convolution: [Float]
            let recurrent: [Float]
        }
        struct Weights: Decodable {
            let qkvProjection: [Float]
            let qkvConvolution: [Float]
            let fbgAProjection: [Float]
            let fBProjection: [Float]
            let gBProjection: [Float]
            let aLog: [Float]
            let dtBias: [Float]
            let outputNorm: [Float]
            let outputProjection: [Float]
        }
        struct Case: Decodable {
            let name: String
            let hiddenSize: Int
            let numHeads: Int
            let headDim: Int
            let convKernelSize: Int
            let lowerBound: Float
            let rmsNormEps: Float
            let prefixLength: Int
            let weights: Weights
            let tokens: [[Float]]
            let steps: [Step]
            let token1Fresh: Step
            let decayMin: Float
            let decayMax: Float
            let decaySpanWithinHead: Float
            let outputGateMin: Float
            let outputGateMax: Float
            let historyGap: Float
        }
        let mlx: String
        let revision: String
        let languageSha256: String
        let gatedDeltaSha256: String
        let cases: [Case]
    }

    private static func load() throws -> Fixture {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures/glm-linear-attention/reference.json")
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Fixture.self, from: Data(contentsOf: url))
    }

    private func make(_ item: Fixture.Case) throws -> GLMNextLinearAttention {
        let configuration = try GLMNextLinearAttention.Configuration(
            hiddenSize: item.hiddenSize,
            numHeads: item.numHeads,
            headDim: item.headDim,
            convKernelSize: item.convKernelSize,
            lowerBound: item.lowerBound,
            rmsNormEps: item.rmsNormEps
        )
        let stored = item.weights
        let weights = try GLMNextLinearAttention.Weights(
            configuration: configuration,
            qkvProjection: stored.qkvProjection,
            qkvConvolution: stored.qkvConvolution,
            fbgAProjection: stored.fbgAProjection,
            fBProjection: stored.fBProjection,
            gBProjection: stored.gBProjection,
            aLog: stored.aLog,
            dtBias: stored.dtBias,
            outputNorm: stored.outputNorm,
            outputProjection: stored.outputProjection
        )
        return try GLMNextLinearAttention(configuration: configuration, weights: weights)
    }

    private func expectClose(_ actual: [Float], _ expected: [Float], tolerance: Float = 1e-6,
                             _ message: String = "values") {
        #expect(actual.count == expected.count, "\(message) count")
        let error = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
        #expect(error <= tolerance, "\(message) max abs error \(error)")
    }

    @Test func fixturePinsTheUpstreamSource() throws {
        let fixture = try Self.load()
        #expect(fixture.mlx == "0.32.2")
        #expect(fixture.revision == "a74c7de90a344a2c2c7334acb4e48b57a40480e2")
        #expect(fixture.languageSha256 == "0ee62da7ea9655fb0a8886fbe30c40f1a5a057aa92960e6f7d1e2ab620d530dd")
        #expect(fixture.gatedDeltaSha256 == "7cac81d20827d0db32fcd8dc2a57a4be21a158b88d826bb1f0da803cf54bafbc")
        #expect(fixture.cases.map(\.name) == ["vector-safe-gate", "kernel-one-safe-gate"])
    }

    @Test func productionDimensionsDoNotRequireWeights() throws {
        let config = try GLMNextLinearAttention.Configuration.glm53FlashText()
        #expect(config.hiddenSize == 4096)
        #expect(config.numHeads == 64)
        #expect(config.headDim == 128)
        #expect(config.convKernelSize == 4)
        #expect(config.lowerBound == -5)
        #expect(config.rmsNormEps == 1e-5)
        #expect(config.queryKeyNormEps == Float(1e-6 / 128.0))
        #expect(config.projectionDim == 8_192)
        #expect(config.qkvChannels == 24_576)
        #expect(config.convolutionStateCount == 73_728)
        #expect(config.recurrentStateCount == 1_048_576)
        #expect(config.qkvProjectionCount == 100_663_296)
        #expect(config.qkvConvolutionCount == 98_304)
        #expect(config.fbgAProjectionCount == 1_310_720)
        #expect(config.fBProjectionCount == 1_048_576)
        #expect(config.gBProjectionCount == 1_048_576)
        #expect(config.dtBiasCount == 8_192)
        #expect(config.outputProjectionCount == 33_554_432)
        let kernelOne = try GLMNextLinearAttention.Configuration(
            hiddenSize: 1, numHeads: 1, headDim: 1, convKernelSize: 1,
            lowerBound: -5, rmsNormEps: 1e-5
        )
        #expect(kernelOne.convolutionStateCount == 0)
        #expect(kernelOne.recurrentStateCount == 1)
    }

    @Test func rejectsUnsupportedConfigurations() throws {
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 0, numHeads: 1, headDim: 1, convKernelSize: 1,
                lowerBound: -5, rmsNormEps: 1e-5)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 4, numHeads: 0, headDim: 2, convKernelSize: 2,
                lowerBound: -5, rmsNormEps: 1e-5)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 4, numHeads: 1, headDim: 0, convKernelSize: 2,
                lowerBound: -5, rmsNormEps: 1e-5)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 4, numHeads: 1, headDim: 2, convKernelSize: 0,
                lowerBound: -5, rmsNormEps: 1e-5)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 4, numHeads: 1, headDim: 2, convKernelSize: 2,
                lowerBound: .nan, rmsNormEps: 1e-5)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 4, numHeads: 1, headDim: 2, convKernelSize: 2,
                lowerBound: .infinity, rmsNormEps: 1e-5)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 4, numHeads: 1, headDim: 2, convKernelSize: 2,
                lowerBound: -5, rmsNormEps: 0)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Configuration(
                hiddenSize: 1, numHeads: Int.max, headDim: 4, convKernelSize: 1,
                lowerBound: -5, rmsNormEps: 1e-5)
        }
    }

    @Test func rejectsMismatchedWeightsTokensAndState() throws {
        let item = try #require(try Self.load().cases.first { $0.name == "vector-safe-gate" })
        let model = try make(item)
        var shortQKV = item.weights.qkvProjection
        shortQKV.removeLast()
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Weights(
                configuration: model.configuration,
                qkvProjection: shortQKV,
                qkvConvolution: item.weights.qkvConvolution,
                fbgAProjection: item.weights.fbgAProjection,
                fBProjection: item.weights.fBProjection,
                gBProjection: item.weights.gBProjection,
                aLog: item.weights.aLog,
                dtBias: item.weights.dtBias,
                outputNorm: item.weights.outputNorm,
                outputProjection: item.weights.outputProjection
            )
        }
        var nonFinite = item.weights.outputNorm
        nonFinite[0] = .nan
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention.Weights(
                configuration: model.configuration,
                qkvProjection: item.weights.qkvProjection,
                qkvConvolution: item.weights.qkvConvolution,
                fbgAProjection: item.weights.fbgAProjection,
                fBProjection: item.weights.fBProjection,
                gBProjection: item.weights.gBProjection,
                aLog: item.weights.aLog,
                dtBias: item.weights.dtBias,
                outputNorm: nonFinite,
                outputProjection: item.weights.outputProjection
            )
        }
        let other = try GLMNextLinearAttention.Configuration(
            hiddenSize: 6, numHeads: 1, headDim: 3, convKernelSize: 1,
            lowerBound: -1.5, rmsNormEps: 1e-5
        )
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try GLMNextLinearAttention(configuration: other, weights: model.weights)
        }
        var state = model.makeState()
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try model.step(Array(item.tokens[0].dropLast()), state: &state)
        }
        #expect(throws: GLMNextLinearAttention.Error.self) {
            var bad = item.tokens[0]
            bad[0] = .infinity
            _ = try model.step(bad, state: &state)
        }
        state.recurrent.append(0)
        #expect(throws: GLMNextLinearAttention.Error.self) {
            _ = try model.step(item.tokens[0], state: &state)
        }
    }

    @Test(arguments: ["vector-safe-gate", "kernel-one-safe-gate"])
    func sequentialStepsMatchMLXStateAndGatedOutput(_ name: String) throws {
        let fixture = try Self.load()
        let item = try #require(fixture.cases.first { $0.name == name })
        if name == "vector-safe-gate" {
            #expect(item.lowerBound == -5)
            #expect(item.convKernelSize == 4)
            #expect(item.decayMin < 0.05)
            #expect(item.decayMax > 0.6)
            #expect(item.decaySpanWithinHead > 0.4)
            #expect(item.outputGateMin < 0.25)
            #expect(item.outputGateMax > 0.75)
        } else {
            #expect(item.lowerBound == -1.5)
            #expect(item.convKernelSize == 1)
            #expect(item.decayMin < 0.4)
            #expect(item.decayMax > 0.8)
            #expect(item.decaySpanWithinHead > 0.35)
            #expect(item.outputGateMin < 0.3)
            #expect(item.outputGateMax > 0.7)
        }
        #expect(item.historyGap > 1e-3)
        #expect(item.steps.count == item.tokens.count)
        let model = try make(item)
        var state = model.makeState()
        for (token, expected) in zip(item.tokens, item.steps) {
            let output = try model.step(token, state: &state)
            expectClose(output, expected.output, "\(name) output")
            expectClose(state.convolution, expected.convolution, "\(name) convolution")
            expectClose(state.recurrent, expected.recurrent, "\(name) recurrent")
        }
        if name == "kernel-one-safe-gate" {
            #expect(item.steps.allSatisfy { $0.convolution.isEmpty })
        }
    }

    @Test(arguments: ["vector-safe-gate", "kernel-one-safe-gate"])
    func continuationMatchesPrefixState(_ name: String) throws {
        let item = try #require(try Self.load().cases.first { $0.name == name })
        let model = try make(item)
        var state = model.makeState()
        for index in 0..<item.prefixLength {
            _ = try model.step(item.tokens[index], state: &state)
        }
        let checkpoint = item.steps[item.prefixLength - 1]
        expectClose(state.convolution, checkpoint.convolution, "\(name) prefix convolution")
        expectClose(state.recurrent, checkpoint.recurrent, "\(name) prefix recurrent")
        for index in item.prefixLength..<item.tokens.count {
            let output = try model.step(item.tokens[index], state: &state)
            expectClose(output, item.steps[index].output, "\(name) continued output")
            expectClose(state.convolution, item.steps[index].convolution, "\(name) continued convolution")
            expectClose(state.recurrent, item.steps[index].recurrent, "\(name) continued recurrent")
        }
    }

    @Test(arguments: ["vector-safe-gate", "kernel-one-safe-gate"])
    func freshTokenDiffersFromContinuedState(_ name: String) throws {
        let item = try #require(try Self.load().cases.first { $0.name == name })
        let gap = zip(item.steps[1].output, item.token1Fresh.output).map { abs($0 - $1) }.max() ?? 0
        #expect(gap > 1e-3)
        let model = try make(item)
        var fresh = model.makeState()
        let output = try model.step(item.tokens[1], state: &fresh)
        expectClose(output, item.token1Fresh.output, "\(name) fresh output")
        expectClose(fresh.convolution, item.token1Fresh.convolution, "\(name) fresh convolution")
        expectClose(fresh.recurrent, item.token1Fresh.recurrent, "\(name) fresh recurrent")
        let recurrentGap = zip(fresh.recurrent, item.steps[1].recurrent).map { abs($0 - $1) }.max() ?? 0
        #expect(recurrentGap > 1e-4)
    }

    @Test func resetRestoresAFreshSequenceAndKeepsAnEarlierCopy() throws {
        let item = try #require(try Self.load().cases.first { $0.name == "vector-safe-gate" })
        let model = try make(item)
        var state = model.makeState()
        for (token, expected) in zip(item.tokens, item.steps) {
            let output = try model.step(token, state: &state)
            expectClose(output, expected.output, "first pass")
        }
        let savedConvolution = Array(state.convolution)
        let savedRecurrent = Array(state.recurrent)
        state.reset()
        #expect(state.convolution.allSatisfy { $0 == 0 })
        #expect(state.recurrent.allSatisfy { $0 == 0 })
        expectClose(savedConvolution, item.steps[item.steps.count - 1].convolution, "saved convolution")
        expectClose(savedRecurrent, item.steps[item.steps.count - 1].recurrent, "saved recurrent")
        #expect(savedRecurrent.contains { $0 != 0 })
        for (token, expected) in zip(item.tokens, item.steps) {
            let output = try model.step(token, state: &state)
            expectClose(output, expected.output, "replay after reset")
            expectClose(state.recurrent, expected.recurrent, "replay recurrent")
        }
    }

    @Test func separateStatesDoNotInterfere() throws {
        let item = try #require(try Self.load().cases.first { $0.name == "vector-safe-gate" })
        let model = try make(item)
        var first = model.makeState()
        var second = model.makeState()
        let firstOutput = try model.step(item.tokens[0], state: &first)
        let secondOutput = try model.step(item.tokens[1], state: &second)
        expectClose(firstOutput, item.steps[0].output, "first state output")
        expectClose(first.recurrent, item.steps[0].recurrent, "first state recurrent")
        expectClose(secondOutput, item.token1Fresh.output, "second state output")
        expectClose(second.recurrent, item.token1Fresh.recurrent, "second state recurrent")
        let continued = try model.step(item.tokens[1], state: &first)
        expectClose(continued, item.steps[1].output, "continued first state")
        expectClose(second.recurrent, item.token1Fresh.recurrent, "untouched second state")
    }
}
