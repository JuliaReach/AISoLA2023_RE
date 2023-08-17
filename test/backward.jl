for dummy in [1]
    #######################
    # POLYHEDRA ALGORITHM #
    #######################

    algo = PolyhedraBackwardAlgorithm()

    # invalid input: incompatible matrix/vector dimensions in affine map
    Y = Singleton([1.0, 1.0])
    @test_throws AssertionError backward(Y, rand(2, 3), rand(1), algo)
    # invalid input: 2D but not a polyhedron or union
    Y = Ball2([-1.0, 1.0], 1.0)
    @test_throws AssertionError backward(Y, ReLU(), algo)

    # 1D ReLU
    Y = BallInf([2.0], 1.0)
    @test backward(Y, ReLU(), algo) == Y
    Y = HalfSpace([2.0], 1.0)
    @test backward(Y, ReLU(), algo) == HalfSpace([1.0], 0.5)
    Y = HalfSpace([-2.0], -1.0)
    @test backward(Y, ReLU(), algo) == HalfSpace([-1.0], -0.5)
    Y = HalfSpace([-2.0], 1.0)
    @test backward(Y, ReLU(), algo) == Universe(1)
    Y = Universe(1)
    @test backward(Y, ReLU(), algo) == Universe(1)

    # 2D ReLU
    Pneg = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 0.0)])
    Px = HPolyhedron([HalfSpace([0.0, 1.0], 0.0), HalfSpace([1.0, 0.0], 2.0),
                      HalfSpace([-1.0, 0.0], -1.0)])
    Py = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 2.0),
                      HalfSpace([0.0, -1.0], -1.0)])
    Qx = HPolyhedron([HalfSpace([0.0, 1.0], 0.0), HalfSpace([1.0, 0.0], 2.0),
                      HalfSpace([-1.0, 0.0], 0.0)])
    Qy = HPolyhedron([HalfSpace([1.0, 0.0], 0.0), HalfSpace([0.0, 1.0], 2.0),
                      HalfSpace([0.0, -1.0], 0.0)])
    # strictly positive
    Y = LineSegment([1.0, 1.0], [2.0, 2.0])
    @test backward(Y, ReLU(), algo) == Y
    # only origin
    Y = Singleton([0.0, 0.0])
    @test backward(Y, ReLU(), algo) == Pneg
    # origin + positive
    Y = LineSegment([0.0, 0.0], [2.0, 2.0])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Pneg])
    # only x-axis
    Y = LineSegment([1.0, 0.0], [2.0, 0.0])
    @test backward(Y, ReLU(), algo) == Px
    # positive + x-axis
    Y = VPolygon([[1.0, 0.0], [2.0, 2.0], [2.0, 0.0]])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Px])
    # only y-axis
    Y = LineSegment([0.0, 1.0], [0.0, 2.0])
    @test backward(Y, ReLU(), algo) == Py
    # positive + y-axis
    Y = VPolygon([[0.0, 1.0], [2.0, 2.0], [0.0, 2.0]])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Py])
    # positive + both axes
    Y = VPolygon([[0.0, 1.0], [0.0, 2.0], [1.0, 0.0], [2.0, 0.0]])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Px, Py])
    # positive + x-axis + origin
    Y = VPolygon([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Qx, Pneg])
    # positive + y-axis + origin
    Y = VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 2.0]])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Qy, Pneg])
    # positive + both axes + origin
    Y = VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y, Qx, Qy, Pneg])
    # strictly negative
    Y = LineSegment([-1.0, -1.0], [-2.0, -2.0])
    @test backward(Y, ReLU(), algo) == EmptySet(2)
    # origin + negative
    Y = LineSegment([0.0, 0.0], [-2.0, -2.0])
    @test backward(Y, ReLU(), algo) == Pneg
    # positive + negative + both axes + origin
    Y = VPolygon([[-1.0, -1.0], [-1.0, 3.0], [3.0, -1.0]])
    X = backward(Y, ReLU(), algo)
    @test X isa UnionSetArray && length(X) == 4 && X[2:4] == [Qx, Qy, Pneg] &&
          isequivalent(X[1], VPolygon([[0.0, 0.0], [0.0, 2.0], [2.0, 0.0]]))
    # union
    Y = UnionSetArray([LineSegment([1.0, 1.0], [2.0, 2.0]), Singleton([0.0, 0.0])])
    @test backward(Y, ReLU(), algo) == UnionSetArray([Y[1], Pneg])

    # 3D ReLU
    # positive point
    Y = Singleton([1.0, 1.0, 1.0])
    @test backward(Y, ReLU(), algo) == Y
    # positive + negative + both axes + origin
    Y = BallInf(zeros(3), 1.0)
    X = backward(Y, ReLU(), algo)  # result: x <= 1 && y <= 1 && z <= 1
    # union is too complex -> only perform sufficient tests
    @test X isa UnionSetArray && length(X) == 8
    @test all(high(X, i) == 1.0 for i in 1:3)
    @test all(low(X, i) == -Inf for i in 1:3)

    # 1D affine map
    X = Singleton([1.0])
    W = hcat([2.0])
    b = [1.0]
    Y = affine_map(W, X, b)
    @test isequivalent(backward(Y, W, b, algo), X)

    # 2D affine map
    X = Singleton([1.0, 2.0])
    W = hcat([2.0 3.0; -1.0 -2.0])
    b = [1.0, -2.0]
    Y = affine_map(W, X, b)
    @test isequivalent(backward(Y, W, b, algo), X)

    # 1D-2D affine map
    X = Singleton([1.0])
    W = hcat([2.0; -1.0])
    b = [1.0, -2.0]
    Y = affine_map(W, X, b)
    @test isequivalent(backward(Y, W, b, algo), X)

    # 2D-1D affine map
    X = Singleton([1.0, 2.0])
    W = hcat([2.0 3.0])
    b = [1.0]
    Y = affine_map(W, X, b)
    X2 = backward(Y, W, b, algo)
    @test X ⊆ X2
    # special case: Interval output
    X = LineSegment([1.0, 2.0], [3.0, 4.0])
    W = hcat([2.0 3.0])
    b = [1.0]
    Y = convert(Interval, affine_map(W, X, b))
    X2 = backward(Y, W, b, algo)
    H1 = HalfSpace([2.0, 3.0], 18.0)
    H2 = HalfSpace([-2.0, -3.0], -8.0)
    @test X ⊆ X2 && (X2.constraints == [H1, H2] || X2.constraints == [H2, H1])
    # special case: HalfSpace output
    Y = HalfSpace([2.0], 38.0)
    @test backward(Y, W, b, algo) == H1
    # special case: Universe output
    Y = Universe(1)
    @test backward(Y, W, b, algo) == Universe(2)

    # 2D network
    net = example_network_222()
    x = [1.0, 2.0]
    Y = Singleton(net(x))
    X = backward(Y, net, algo)
    @test x ∈ X
    x = [-4.0, 0.0]
    @test Singleton(net(x)) == Y && x ∈ X

    # 1D/2D network
    net = read_NNet(joinpath(@__DIR__, "small_nnet.nnet"))
    X = Interval(2.5, 5.0)
    Y = convert(Interval, forward(X, net, ConcreteForwardAlgorithm()))
    X2 = backward(Y, net, algo)
    @test isequivalent(X, X2)

    #################
    # BOX ALGORITHM #
    #################

    algo = BoxBackwardAlgorithm()

    # 2D network
    net = example_network_222()
    x = [1.0, 2.0]
    Y = Singleton(net(x))
    X = backward(Y, net, algo)
    @test x ∈ X
    x = [-4.0, 0.0]
    @test Singleton(net(x)) == Y && x ∈ X

    # 1D/2D network
    net = read_NNet(joinpath(@__DIR__, "small_nnet.nnet"))
    X = Interval(2.5, 5.0)
    Y = convert(Interval, forward(X, net, ConcreteForwardAlgorithm()))
    X2 = backward(Y, net, algo)
    @test X ⊆ X2
end
