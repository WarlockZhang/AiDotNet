using AiDotNet.Enums;
using AiDotNet.Interfaces;
using AiDotNet.NeuralNetworks;
using AiDotNet.NeuralNetworks.Layers;
using Xunit;

namespace AiDotNet.Tests.IntegrationTests.NeuralNetworks;

/// <summary>
/// Tests that ParameterBuffer view replacement is scoped to the training step
/// and does not permanently replace layer tensor references (fixes #1084).
/// </summary>
public class ParameterBufferScopeTests
{
    private static FeedForwardNeuralNetwork<double> CreateSimpleNetwork()
    {
        var arch = new NeuralNetworkArchitecture<double>(
            inputType: InputType.OneDimensional,
            taskType: NeuralNetworkTaskType.Regression,
            inputSize: 4,
            outputSize: 2);
        return new FeedForwardNeuralNetwork<double>(arch);
    }

    [Fact]
    public void Train_DoesNotPermanentlyReplaceLayerTensors()
    {
        var network = CreateSimpleNetwork();

        // Capture original tensor references before training
        var originalParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                    originalParams.Add(p);
            }
        }

        // Train one step
        var input = Tensor<double>.CreateRandom([1, 4]);
        var target = Tensor<double>.CreateRandom([1, 2]);
        network.Train(input, target);

        // Tensor references should be the same objects (not buffer views)
        int idx = 0;
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                {
                    if (idx < originalParams.Count)
                    {
                        Assert.True(ReferenceEquals(p, originalParams[idx]),
                            $"Parameter {idx} was permanently replaced with a buffer view");
                    }
                    idx++;
                }
            }
        }
    }

    [Fact]
    public void Train_UpdatesParameterValues()
    {
        var network = CreateSimpleNetwork();

        // Snapshot ALL parameter data before training
        var beforeSnapshot = new Dictionary<int, double[]>();
        int tensorIdx = 0;
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                {
                    var data = new double[p.Length];
                    p.AsSpan().CopyTo(data);
                    beforeSnapshot[tensorIdx++] = data;
                }
            }
        }

        Assert.True(beforeSnapshot.Count > 0, "Network should have trainable parameters");

        // Train — use larger values to ensure non-zero gradients
        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;

        // Test tape with DenseLayer directly
        var denseLayer = new DenseLayer<double>(4, 2);
        using (var testTape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<double>())
        {
            var testOutput = denseLayer.Forward(input);
            var denseEntries = testTape.EntryCount;
            if (denseEntries == 0)
                throw new Exception($"DenseLayer tape test: Forward recorded 0 entries! output={testOutput[0,0]:F6}");
        }

        for (int step = 0; step < 10; step++)
            network.Train(input, target);

        // Compare ALL parameter data after training
        tensorIdx = 0;
        bool anyChanged = false;
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                {
                    var before = beforeSnapshot[tensorIdx++];
                    for (int i = 0; i < p.Length; i++)
                    {
                        if (Math.Abs(before[i] - p.GetFlat(i)) > 1e-12)
                        {
                            anyChanged = true;
                            break;
                        }
                    }
                    if (anyChanged) break;
                }
                if (anyChanged) break;
            }
        }

        // Debug: show actual values
        tensorIdx = 0;
        var debugBefore = new List<string>();
        var debugAfter = new List<string>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                {
                    var before = beforeSnapshot[tensorIdx];
                    debugBefore.Add($"t{tensorIdx}[0]={before[0]:F8} len={before.Length}");
                    debugAfter.Add($"t{tensorIdx}[0]={p.GetFlat(0):F8} len={p.Length}");
                    tensorIdx++;
                }
            }
        }

        if (!anyChanged)
        {
            throw new Exception(
                $"Training 10 steps should update parameters.\n" +
                $"ParamTensors: {beforeSnapshot.Count}\n" +
                $"GradsReturned: {network._lastGradientCount}\n" +
                $"ParamsCollected: {network._lastParameterCount}\n" +
                $"TapeEntries: {network._lastTapeEntryCount}\n" +
                $"TapeActive: {network._lastTapeWasActive}\n" +
                $"LossLen: {network._lastLossLength}\n" +
                $"LossVal: {network._lastLossValue:F8}\n" +
                $"ForwardTapeActive: {network._forwardTapeActive}\n" +
                $"ForwardTapeEntriesAfter: {network._forwardTapeEntriesAfter}\n" +
                $"NonZeroGrads: {network._lastNonZeroGradCount}\n" +
                $"LossHasGradFn: {network._lastLossHasGradFn}\n" +
                $"ParamViewMatch: {network._lastParamViewMatch}\n" +
                $"AllGradsCount: {network._lastAllGradsCount}\n" +
                $"Before: {string.Join(" | ", debugBefore)}\n" +
                $"After:  {string.Join(" | ", debugAfter)}");
        }
    }
}
