for dummy in [1]
    algoB = BoxBidirectionalAlgorithm()
    algoP = PolyhedraBidirectionalAlgorithm()
    algoBP = SimpleBidirectionalAlgorithm(ConcreteForwardAlgorithm(), BoxBackwardAlgorithm())
    algoPB = SimpleBidirectionalAlgorithm(BoxForwardAlgorithm(), PolyhedraBackwardAlgorithm())

    net = example_network_222()
    x = [1.0, 1.0]
    @test forward(x, net) == [-5.0, 5.0]

    # initial point
    X = Singleton(x)
    # empty preimage
    Y = Singleton([5.0, 5.0])
    for algo in (algoB, algoP, algoBP, algoPB)
        X2, Y2 = bidirectional(X, Y, net, algo)
        @test X2 == EmptySet(2)
    end
    # nonempty preimage
    Y = HalfSpace([1.0, -1.0], 0.0)
    for algo in (algoB, algoP, algoBP, algoPB)
        X2, Y2 = bidirectional(X, Y, net, algo)
        @test !isdisjoint(Y, Y2) && isequivalent(X, X2)
    end

    # initial set
    X = BallInf(x, 1.0)
    # empty preimage
    Y = Singleton([5.0, 5.0])
    for algo in (algoB, algoP, algoBP, algoPB)
        X2, Y2 = bidirectional(X, Y, net, algo)
        @test X2 == EmptySet(2)
    end
    # nonempty preimage
    Y = HalfSpace([1.0, -1.0], 0.0)
    for algo in (algoB, algoP, algoBP, algoPB)
        @test_broken X2, Y2 = bidirectional(X, Y, net, algo)
        @test_broken !isdisjoint(Y, Y2) && isequivalent(X, X2)  # requires LazySets#3350
    end
end
