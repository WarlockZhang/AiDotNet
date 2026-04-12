using AiDotNet.Data;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNetTests.Data;

/// <summary>
/// Tests for the ImageTensorLayout option on vision data loaders.
/// Uses AutoDownload=false and SkippableFact — reports as SKIP (not PASS)
/// when datasets aren't cached, so CI output clearly shows missing coverage.
/// </summary>
public class ImageTensorLayoutTests
{
    private static bool MnistCacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("mnist");
        return File.Exists(Path.Combine(cachePath, "train-images-idx3-ubyte"))
            && File.Exists(Path.Combine(cachePath, "train-labels-idx1-ubyte"));
    }

    private static bool Cifar10CacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("cifar-10");
        string batchesDir = Path.Combine(cachePath, "cifar-10-batches-bin");
        return (Directory.Exists(cachePath) && Directory.GetFiles(cachePath, "data_batch*").Length > 0)
            || (Directory.Exists(batchesDir) && Directory.GetFiles(batchesDir, "data_batch*").Length > 0);
    }

    [SkippableFact]
    public async Task MnistNHWC_DefaultShape_Is_B28281()
    {
        Skip.IfNot(MnistCacheExists(), "MNIST not cached locally");
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        Assert.Equal(new[] { 2, 28, 28, 1 }, batch.Features.Shape.ToArray());
    }

    [SkippableFact]
    public async Task MnistNCHW_Shape_Is_B1_28_28()
    {
        Skip.IfNot(MnistCacheExists(), "MNIST not cached locally");
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        Assert.Equal(new[] { 2, 1, 28, 28 }, batch.Features.Shape.ToArray());
    }

    [SkippableFact]
    public async Task MnistFlatten_IgnoresLayout()
    {
        Skip.IfNot(MnistCacheExists(), "MNIST not cached locally");
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 2,
            Flatten = true, Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        Assert.Equal(new[] { 2, 784 }, batch.Features.Shape.ToArray());
    }

    [SkippableFact]
    public async Task Cifar10NCHW_ShapeAndChannelOrder()
    {
        Skip.IfNot(Cifar10CacheExists(), "CIFAR-10 not cached locally");
        var loader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
        {
            Split = DatasetSplit.Train, AutoDownload = false, MaxSamples = 1,
            Normalize = false, Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 1).First();
        Assert.Equal(new[] { 1, 3, 32, 32 }, batch.Features.Shape.ToArray());
    }
}
