using AiDotNet.Interfaces;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Microsoft.Extensions.Logging;

namespace AiDotNet.Training;

/// <summary>
/// Compiled training step — auto-compiles the forward + backward pass on the first step,
/// then replays the compiled plan on subsequent steps for near-zero overhead training.
///
/// <para><b>How it works:</b></para>
/// <list type="number">
/// <item><b>Step 1 (tracing):</b> Enables GraphMode, traces the forward pass + loss computation
/// through the layer stack, compiles a CompiledTrainingPlan with backward pass, and executes it.</item>
/// <item><b>Steps 2+ (replay):</b> Calls plan.Step() which replays the compiled forward + backward
/// as flat delegate arrays with pre-allocated gradient buffers. Zero allocation, zero dispatch overhead.</item>
/// </list>
///
/// <para><b>Recompilation triggers:</b></para>
/// <list type="bullet">
/// <item>Input shape changes (different batch size, sequence length, etc.)</item>
/// <item>Explicit Invalidate() call (model structure changed)</item>
/// <item>Compilation failure (falls back to eager TapeTrainingStep for that shape)</item>
/// </list>
/// </summary>
/// <typeparam name="T">The numeric type.</typeparam>
public static class CompiledTapeTrainingStep<T>
{
    [ThreadStatic]
    private static CompiledModelCache<T>? _cache;
    [ThreadStatic]
    private static int[]? _lastInputShape;
    [ThreadStatic]
    private static Tensor<T>[]? _cachedParameters;

    private static readonly ILogger? _logger = LoggerFactory.Create(b => { }).CreateLogger(typeof(CompiledTapeTrainingStep<T>).Name);

    /// <summary>
    /// Executes a single compiled training step.
    /// First call traces and compiles; subsequent calls replay the compiled plan.
    /// Falls back to eager execution if compilation fails.
    /// </summary>
    /// <param name="layers">Trainable layers of the model.</param>
    /// <param name="input">Input tensor for this batch.</param>
    /// <param name="target">Target tensor for loss computation.</param>
    /// <param name="learningRate">SGD learning rate.</param>
    /// <param name="forward">Forward pass function: input -> prediction.</param>
    /// <param name="computeLoss">Loss function: (prediction, target) -> loss scalar.</param>
    /// <returns>The scalar loss value for this step.</returns>
    public static T Step(
        IReadOnlyList<ITrainableLayer<T>> layers,
        Tensor<T> input,
        Tensor<T> target,
        T learningRate,
        Func<Tensor<T>, Tensor<T>> forward,
        Func<Tensor<T>, Tensor<T>, Tensor<T>> computeLoss)
    {
        if (!TensorCodecOptions.Current.EnableCompilation)
            return TapeTrainingStep<T>.Step(layers, input, target, learningRate, forward, computeLoss);

        var numOps = MathHelper.GetNumericOperations<T>();
        var engine = AiDotNetEngine.Current;

        try
        {
            var cache = _cache ??= new CompiledModelCache<T>();

            // Detect shape change — triggers recompilation and parameter re-collection
            bool shapeChanged = !ShapeMatches(input._shape, _lastInputShape);
            if (shapeChanged)
            {
                _lastInputShape = input._shape;
                _cachedParameters = null;
            }

            // Force layer initialization before collecting parameters.
            // DenseLayer.EnsureInitialized() replaces _weights with a new tensor on
            // first Forward — collecting before that captures stale placeholder tensors.
            // A dry-run forward triggers initialization without GraphMode recording.
            if (_cachedParameters is null)
            {
                forward(input);
            }

            // Now safe to collect — layers are initialized, tensors are final
            var parameters = _cachedParameters ??= CollectParameterArray(layers);

            // Zero gradients before forward pass
            foreach (var layer in layers)
                layer.ZeroGrad();

            // Get or compile training plan (cached by shape)
            var plan = cache.GetOrCompileTraining(
                input._shape,
                () =>
                {
                    var predicted = forward(input);
                    computeLoss(predicted, target);
                },
                parameters);

            // Execute compiled forward + backward
            var lossOutput = plan.Step();

            // Update parameters with SGD
            UpdateParametersSGD(engine, parameters, plan.Gradients, learningRate, numOps);

            return lossOutput.Length > 0 ? lossOutput[0] : numOps.Zero;
        }
        catch (Exception ex)
        {
            // Log the compilation failure for developer diagnostics
            _logger?.LogWarning(ex, "Compiled training step failed, falling back to eager execution");

            // Fall back to eager for this step only — next step will retry compilation.
            // Don't permanently disable compilation; the failure may be transient
            // (e.g., unsupported op that gets fixed in a later version).
            return TapeTrainingStep<T>.Step(layers, input, target, learningRate, forward, computeLoss);
        }
    }

    /// <summary>
    /// Invalidates the compiled plan cache. Call when model structure changes
    /// (layers added/removed, activation functions changed, etc.).
    /// </summary>
    public static void Invalidate()
    {
        _cache?.Invalidate();
        _lastInputShape = null;
        _cachedParameters = null; // Force parameter re-collection
    }

    private static Tensor<T>[] CollectParameterArray(IReadOnlyList<ITrainableLayer<T>> layers)
    {
        var allParams = new List<Tensor<T>>();
        foreach (var layer in layers)
            allParams.AddRange(layer.GetTrainableParameters());
        return allParams.ToArray();
    }

    private static void UpdateParametersSGD(
        IEngine engine, Tensor<T>[] parameters, Tensor<T>[] gradients,
        T learningRate, INumericOperations<T> numOps)
    {
        int count = Math.Min(parameters.Length, gradients.Length);
        for (int i = 0; i < count; i++)
        {
            if (gradients[i] is not null)
            {
                var update = engine.TensorMultiplyScalar(gradients[i], learningRate);
                engine.TensorSubtractInPlace(parameters[i], update);
            }
        }
    }

    private static bool ShapeMatches(int[] a, int[]? b)
    {
        if (b is null || a.Length != b.Length) return false;
        // Reference equality first — same tensor reuses the same _shape array
        if (ReferenceEquals(a, b)) return true;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }
}
