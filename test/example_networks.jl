function example_network_222()
    return FeedforwardNetwork([DenseLayerOp([1.0 2.0; -1.0 -2.0], [1.0, -1.0], ReLU()),
                               DenseLayerOp([-1.0 -2.0; 1.0 2.0], [-1.0, 1.0], Id())])
end

function example_network_232()
    return FeedforwardNetwork([DenseLayerOp([1.0 2.0; -1.0 -2.0; 3.0 -3.0], [1.0, -1.0, 1.0],
                                            ReLU()),
                               DenseLayerOp([-1.0 -2.0 -3.0; 1.0 2.0 3.0], [-1.0, 1.0], Id())])
end

function example_network_1221()
    return read_NNet(joinpath(@__DIR__, "small_nnet.nnet"))
end
