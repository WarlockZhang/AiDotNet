using AiDotNet.Data;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNetTests.Data;

/// <summary>
/// Tests for the ImageTensorLayout option on vision data loaders.
/// Uses AutoDownload=false + skips if MNIST isn't cached locally,
/// so these tests never hit the network and are CI-safe.
/// </summary>
public class ImageTensorLayoutTests
{
    private static bool MnistCacheExists()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("mnist");
        return File.Exists(Path.Combine(cachePath, "train-images-idx3-ubyte"));
    }

    [Fact]
    public async Task MnistNHWC_DefaultShape_Is_B28281()
    {
        if (!MnistCacheExists()) return;
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = false,
            MaxSamples = 2,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        Assert.Equal(new[] { 2, 28, 28, 1 }, batch.Features.Shape.ToArray());
    }

    [Fact]
    public async Task MnistNCHW_Shape_Is_B1_28_28()
    {
        if (!MnistCacheExists()) return;
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = false,
            MaxSamples = 2,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        Assert.Equal(new[] { 2, 1, 28, 28 }, batch.Features.Shape.ToArray());
    }

    [Fact]
    public async Task MnistFlatten_IgnoresLayout()
    {
        if (!MnistCacheExists()) return;
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = false,
            MaxSamples = 2,
            Flatten = true,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        Assert.Equal(new[] { 2, 784 }, batch.Features.Shape.ToArray());
    }
}
