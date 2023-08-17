for (N, y) in [(example_network_222(), [-5.0, 5.0]),
               (example_network_232(), [-8.0, 8.0])]
    x = Singleton([1.0, 1.0])

    ys = forward(element(x), N)
    y1 = forward(x, N)
    y2 = forward(x, N, ConcreteForwardAlgorithm())
    y3 = forward(x, N, LazyForwardAlgorithm())
    y4 = forward(x, N, BoxForwardAlgorithm())
    y5 = forward(x, N, BoxForwardAlgorithm(LazyForwardAlgorithm()))
    @test ys == y
    for yi in (y1, y2, y3, y4, y5)
        @test isequivalent(concretize(yi), Singleton(y))
    end

    ys = forward_all(element(x), N)
    y1 = forward_all(x, N)
    y2 = forward_all(x, N, ConcreteForwardAlgorithm())
    y3 = forward_all(x, N, LazyForwardAlgorithm())
    y4 = forward_all(x, N, BoxForwardAlgorithm())
    y5 = forward_all(x, N, BoxForwardAlgorithm(LazyForwardAlgorithm()))
    @test ys[end][2] == y
    for yi in (y1, y2, y3, y4, y5)
        @test isequivalent(concretize(yi[end][2]), Singleton(y))
    end
end
