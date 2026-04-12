using AiDotNet.Data.Geometry;
using AiDotNet.Data.Vision.Benchmarks;
using Xunit;

namespace AiDotNetTests.Data;

public class ImageTensorLayoutTests
{
    [Fact]
    public async Task MnistNHWC_DefaultShape_Is_B28281()
    {
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = true,
            MaxSamples = 2,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        var shape = batch.Features.Shape.ToArray();
        Assert.Equal(4, shape.Length);
        Assert.Equal(2, shape[0]);
        Assert.Equal(28, shape[1]);
        Assert.Equal(28, shape[2]);
        Assert.Equal(1, shape[3]);
    }

    [Fact]
    public async Task MnistNCHW_Shape_Is_B1_28_28()
    {
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = true,
            MaxSamples = 2,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        var shape = batch.Features.Shape.ToArray();
        Assert.Equal(4, shape.Length);
        Assert.Equal(2, shape[0]);
        Assert.Equal(1, shape[1]);
        Assert.Equal(28, shape[2]);
        Assert.Equal(28, shape[3]);
    }

    [Fact]
    public async Task MnistFlatten_IgnoresLayout()
    {
        var loader = new MnistDataLoader<float>(new MnistDataLoaderOptions
        {
            Split = DatasetSplit.Train,
            AutoDownload = true,
            MaxSamples = 2,
            Flatten = true,
            Layout = ImageTensorLayout.NCHW,
        });
        await loader.LoadAsync();
        var batch = loader.GetBatches(batchSize: 2).First();
        var shape = batch.Features.Shape.ToArray();
        Assert.Equal(2, shape.Length);
        Assert.Equal(2, shape[0]);
        Assert.Equal(784, shape[1]);
    }
}
