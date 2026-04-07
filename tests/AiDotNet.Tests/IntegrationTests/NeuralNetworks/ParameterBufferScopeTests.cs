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
        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;
        network.Train(input, target);

        // Collect post-training parameters
        var afterParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
            {
                foreach (var p in trainable.GetTrainableParameters())
                    afterParams.Add(p);
            }
        }

        // Same count
        Assert.Equal(originalParams.Count, afterParams.Count);

        // Same tensor references (not buffer views)
        for (int i = 0; i < originalParams.Count; i++)
        {
            Assert.True(ReferenceEquals(afterParams[i], originalParams[i]),
                $"Parameter {i} was permanently replaced with a buffer view after training");
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

        // Train with non-trivial loss
        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;

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

        Assert.True(anyChanged, "Training 10 steps with non-trivial loss should update parameter values");
    }
}
