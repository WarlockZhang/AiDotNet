using AiDotNet.Data;
using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNetTests.Data;

/// <summary>
/// Tests for the ImageTensorLayout option on vision data loaders.
/// Uses AutoDownload=false; logs explicit SKIP when data isn't cached.
/// </summary>
public class ImageTensorLayoutTests
{
    private readonly ITestOutputHelper _output;

    public ImageTensorLayoutTests(ITestOutputHelper output) => _output = output;

    private bool RequireMnistCache()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("mnist");
        if (File.Exists(Path.Combine(cachePath, "train-images-idx3-ubyte")))
            return true;
        _output.WriteLine("SKIPPED: MNIST not cached. Run once with AutoDownload=true to populate.");
        return false;
    }

    [Fact]
    public async Task MnistNHWC_DefaultShape_Is_B28281()
    {
        if (!RequireMnistCache()) return;
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
        if (!RequireMnistCache()) return;
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
        if (!RequireMnistCache()) return;
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

    private bool RequireCifar10Cache()
    {
        string cachePath = DatasetDownloader.GetDefaultDataPath("cifar-10");
        string batchesDir = Path.Combine(cachePath, "cifar-10-batches-bin");
        bool exists = (Directory.Exists(cachePath) && Directory.GetFiles(cachePath, "data_batch*").Length > 0)
            || (Directory.Exists(batchesDir) && Directory.GetFiles(batchesDir, "data_batch*").Length > 0);
        if (!exists)
            _output.WriteLine("SKIPPED: CIFAR-10 not cached locally.");
        return exists;
    }

    [Fact]
    public async Task Cifar10NCHW_ShapeAndChannelOrder()
    {
        if (!RequireCifar10Cache()) return;
        var loader = new Cifar10DataLoader<float>(new Cifar10DataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = false,
            MaxSamples = 2,
            Normalize = false,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        Assert.Equal(new[] { 2, 3, 32, 32 }, batch.Features.Shape.ToArray());
    }
}
