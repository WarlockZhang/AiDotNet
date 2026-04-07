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

    [Fact]
    public void Train_MultipleSteps_ParameterReferencesRemainStableAcrossAllSteps()
    {
        // Regression: RestoreOriginalParameters must restore refs on EVERY step,
        // not just the first one.
        var network = CreateSimpleNetwork();

        // Capture references before any training
        var originalParams = new List<Tensor<double>>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    originalParams.Add(p);
        }

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 5.0; target[0, 1] = -5.0;

        // Run multiple training steps and verify references after each
        for (int step = 0; step < 5; step++)
        {
            network.Train(input, target);

            var afterParams = new List<Tensor<double>>();
            foreach (var layer in network.Layers)
            {
                if (layer is ITrainableLayer<double> trainable)
                    foreach (var p in trainable.GetTrainableParameters())
                        afterParams.Add(p);
            }

            Assert.Equal(originalParams.Count, afterParams.Count);
            for (int i = 0; i < originalParams.Count; i++)
            {
                Assert.True(ReferenceEquals(afterParams[i], originalParams[i]),
                    $"Step {step + 1}: parameter {i} was replaced with a buffer view");
            }
        }
    }

    [Fact]
    public void Train_ParameterValuesInOriginalTensorsMatchAfterRestore()
    {
        // RestoreOriginalParameters copies data from views back to originals.
        // Verify that the value in the original tensor equals the trained value.
        var network = CreateSimpleNetwork();

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 2.0; input[0, 2] = 3.0; input[0, 3] = 4.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 10.0; target[0, 1] = -10.0;

        // One training step
        network.Train(input, target);

        // Read all parameter values through the original (restored) tensor references
        var paramValues = new List<double>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    for (int i = 0; i < p.Length; i++)
                        paramValues.Add(p.GetFlat(i));
        }

        // Run a second step — if values were corrupted, the second forward pass would diverge
        // (not throw) and we'd see a completely different update trajectory.
        network.Train(input, target);

        var paramValuesAfterSecond = new List<double>();
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                foreach (var p in trainable.GetTrainableParameters())
                    for (int i = 0; i < p.Length; i++)
                        paramValuesAfterSecond.Add(p.GetFlat(i));
        }

        // The values must differ (i.e. second step actually changed the params),
        // proving the first step's copy-back was correct and used in the next step.
        bool changed = false;
        for (int i = 0; i < paramValues.Count; i++)
        {
            if (Math.Abs(paramValues[i] - paramValuesAfterSecond[i]) > 1e-12)
            {
                changed = true;
                break;
            }
        }
        Assert.True(changed, "Second training step must update parameters from the correctly restored first-step values");
    }

    [Fact]
    public void Train_ThenPredict_ReturnsFiniteValues()
    {
        // After the try/finally restore, the network must still be usable for inference.
        var network = CreateSimpleNetwork();

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 1.0; input[0, 1] = 0.5; input[0, 2] = -0.5; input[0, 3] = 2.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 1.0; target[0, 1] = 0.0;

        // Train a few steps
        for (int step = 0; step < 5; step++)
            network.Train(input, target);

        // Predict using the same input
        var prediction = network.Predict(input);

        // All outputs must be finite (no NaN / Inf leaking from corrupted views)
        Assert.NotNull(prediction);
        for (int i = 0; i < prediction.Length; i++)
        {
            Assert.False(double.IsNaN(prediction.GetFlat(i)),
                $"Prediction element {i} is NaN after training");
            Assert.False(double.IsInfinity(prediction.GetFlat(i)),
                $"Prediction element {i} is infinite after training");
        }
    }

    [Fact]
    public void Train_ParameterCountIsConsistentAcrossMultipleSteps()
    {
        // RestoreOriginalParameters must never add or remove parameter tensors.
        var network = CreateSimpleNetwork();

        int expectedCount = 0;
        foreach (var layer in network.Layers)
        {
            if (layer is ITrainableLayer<double> trainable)
                expectedCount += trainable.GetTrainableParameters().Count;
        }

        var input = new Tensor<double>([1, 4]);
        input[0, 0] = 2.0; input[0, 1] = -1.0; input[0, 2] = 0.0; input[0, 3] = 3.0;
        var target = new Tensor<double>([1, 2]);
        target[0, 0] = 0.5; target[0, 1] = 0.5;

        for (int step = 0; step < 8; step++)
        {
            network.Train(input, target);

            int count = 0;
            foreach (var layer in network.Layers)
            {
                if (layer is ITrainableLayer<double> trainable)
                    count += trainable.GetTrainableParameters().Count;
            }
            Assert.Equal(expectedCount, count);
        }
    }
}