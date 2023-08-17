using LazySets.Approximations: DIR_EAST, DIR_WEST, DIR_SOUTH, DIR_NORTH
using LazySets: SingleEntryVector, _leq, affine_map_inverse

export backward,
       PolyhedraBackwardAlgorithm,
       BoxBackwardAlgorithm

abstract type BackwardAlgorithm end

remove_constraints(::BackwardAlgorithm) = false

simplify_set(X::EmptySet) = X
simplify_set(X::LazySet{N}) where {N} = isempty(X) ? EmptySet{N}(dim(X)) : X

function simplify_union(sets::AbstractVector; n=1, N=Float64)
    if length(sets) > 1
        return UnionSetArray([X for X in sets])  # allocate set-specific array
    elseif length(sets) == 1
        return sets[1]
    else
        return EmptySet{N}(n)
    end
end

append_sets!(Xs, X::LazySet) = push!(Xs, X)
append_sets!(Xs, X::UnionSetArray) = append!(Xs, array(X))

# d is a vector representing a diagonal matrix
function linear_map_inverse(d::AbstractVector{<:Number}, P::LazySet)
    constraints_P = constraints_list(P)
    constraints_MP = LazySets._preallocate_constraints(constraints_P)
    has_undefs = false
    N = promote_type(eltype(d), eltype(P))
    @inbounds for (i, c) in enumerate(constraints_P)
        cinv = _linear_map_inverse_mult(d, c.a)
        if iszero(cinv)
            if zero(N) <= c.b
                # constraint is redundant
                has_undefs = true
            else
                # constraint is infeasible
                return EmptySet{N}(length(cinv))
            end
        else
            constraints_MP[i] = HalfSpace(cinv, c.b)
        end
    end
    if has_undefs  # there were redundant constraints, so remove them
        constraints_MP = [constraints_MP[i]
                          for i in 1:length(constraints_MP)
                          if isassigned(constraints_MP, i)]
    end
    if isempty(constraints_MP)
        return Universe{N}(size(A, 2))
    end
    return HPolyhedron(constraints_MP)
end

function _linear_map_inverse_mult(d::AbstractVector{<:Number}, a)
    return [d[i] * a[i] for i in eachindex(a)]
end

function _linear_map_inverse_mult(d::AbstractVector{Bool}, a)
    return [d[i] ? a[i] : zero(eltype(a)) for i in eachindex(a)]
end

remove_constraints!(::LazySet) = nothing

function remove_constraints!(P::LazySets.HPoly)
    m1 = length(P.constraints)
    remove_redundant_constraints!(P)
    m2 = length(P.constraints)
    return println("$(m2 - m1)/$m1 constraints removed")
end

#####################
# default algorithm #
#####################

# backpropagate y through network
function backward(y, net::FeedforwardNetwork, algo::BackwardAlgorithm)
    # backpropagation
    @inbounds for L in reverse(net.layers)
        y = backward(y, L, algo)

        # early termination check
        if y isa EmptySet
            y = EmptySet(dim_in(net))
            break
        end
    end

    # output
    X = y
    if X isa UnionSetArray && length(X.array) == 1
        return X.array[1]
    else
        return X
    end
end

# backpropagate y through layer
function backward(y, L::DenseLayerOp, algo::BackwardAlgorithm)
    x = backward(y, L.activation, algo)  # apply inverse activation function
    remove_constraints(algo) && remove_constraints!(x)
    if x isa EmptySet
        return x
    end
    w = backward(x, L.weights, L.bias, algo)  # apply inverse affine map
    remove_constraints(algo) && remove_constraints!(w)
    return w
end

# apply inverse affine map to Y
function backward(Y, W::AbstractMatrix, b::AbstractVector, ::BackwardAlgorithm)
    return _backward_affine_map(W, Y, b)
end

# try to invert the matrix
function _backward_affine_map(W::AbstractMatrix, y::AbstractVector, b::AbstractVector)
    return inv(W) * (y .- b)
end

function _backward_affine_map(W::AbstractMatrix, Y::LazySet, b::AbstractVector)
    X = affine_map_inverse(W, Y, b)
    return simplify_set(X)
end

# apply inverse affine map to a union of sets
function backward(Y::UnionSetArray, W::AbstractMatrix, b::AbstractVector,
                  ::BackwardAlgorithm)
    return _backward_union(Y, W, b, algo)
end

function _backward_union(Y::UnionSetArray{N}, W::AbstractMatrix,
                         b::AbstractVector, algo::BackwardAlgorithm) where {N}
    @assert dim(Y) == size(W, 1) == length(b)
    out = []
    for Yi in array(Y)
        append_sets!(out, backward(Yi, W, b, algo))
    end
    filter!(!isempty, out)
    return simplify_union(out; n=size(W, 2), N=N)
end

# apply inverse piecewise-affine activation function to a union of sets
for T in (:ReLU, :LeakyReLU)
    @eval begin
        function backward(Y::UnionSetArray, act::$T, algo::BackwardAlgorithm)
            return _backward_union(Y, act, algo)
        end
    end
end

struct PartitioningLeakyReLU{N<:Real}
    n::Int
    slope::N
end

function Base.length(it::PartitioningLeakyReLU)
    return 2^it.n
end

function Base.iterate(it::PartitioningLeakyReLU{N}, state=nothing) where {N}
    if isnothing(state)
        bv_it = BitvectorIterator(falses(it.n), trues(it.n), true)
        bv, bv_state = iterate(bv_it)
    else
        bv_it, bv_state = state
        bv_res = iterate(bv_it, bv_state)
        if isnothing(bv_res)
            return nothing
        end
        bv, bv_state = bv_res
    end
    state = isnothing(bv_state) ? nothing : (bv_it, bv_state)
    P = _pwa_partition_LeakyReLU(bv, N)
    v = [bv[i] ? one(N) : it.slope for i in eachindex(bv)]
    αA = Diagonal(v)
    αb = zeros(N, length(v))
    res = (P, (αA, αb))
    return (res, state)
end

function _pwa_partition_LeakyReLU(bv, N)
    return HPolyhedron([HalfSpace(SingleEntryVector(i, length(bv), bv[i] ? -one(N) : one(N)),
                                  zero(N)) for i in eachindex(bv)])
end

function pwa_partitioning(::ReLU, n::Int, N)
    return PartitioningLeakyReLU(n, zero(N))
end

function pwa_partitioning(act::LeakyReLU, n::Int, N)
    return PartitioningLeakyReLU(n, N(act.slope))
end

function _backward_union(Y::LazySet{N}, act::ActivationFunction,
                         algo::BackwardAlgorithm) where {N}
    out = []
    for Yi in array(Y)
        Xs = backward(Yi, act, algo)
        if !(Xs isa EmptySet)
            append_sets!(out, Xs)
        end
    end
    return simplify_union(out; n=dim(Y), N=N)
end

function backward(y::AbstractVector, act::ActivationFunction, algo::BackwardAlgorithm)
    return _inverse(y, act)
end

_inverse(x::AbstractVector, act::ActivationFunction) = [_inverse(xi, act) for xi in x]
_inverse(x::Number, ::Id) = x
_inverse(x::Number, ::Sigmoid) = @. -log(1 / x - 1)
_inverse(x::Number, ::ReLU) = x >= zero(x) ? x : zero(x)
_inverse(x::Number, act::LeakyReLU) = x >= zero(x) ? x : x / act.slope

# apply inverse identity activation function to Y
function backward(Y::LazySet, ::Id, ::BackwardAlgorithm)
    return Y
end

# disambiguation
function backward(Y::UnionSetArray, ::Id, ::BackwardAlgorithm)
    return Y
end

################################
# union-of-polyhedra algorithm #
################################

struct PolyhedraBackwardAlgorithm <: BackwardAlgorithm end

remove_constraints(::PolyhedraBackwardAlgorithm) = true

# apply inverse affine map to Y
function backward(Y::LazySet, W::AbstractMatrix, b::AbstractVector,
                  algo::PolyhedraBackwardAlgorithm)
    m = size(W, 1)
    @assert m == dim(Y)
    if m == 1
        X = _backward_1D(Y, W, b, algo)
    else
        X = _backward_nD(Y, W, b, algo)
    end
    return simplify_set(X)
end

# apply inverse affine map to one-dimensional Y
function _backward_1D(Y::LazySet, W::AbstractMatrix, b::AbstractVector,
                      algo::PolyhedraBackwardAlgorithm)
    return _backward_nD(Y, W, b, algo)  # fall back to general method
end

# specialization for polytopes
function _backward_1D(Y::AbstractPolytope, W::AbstractMatrix, b::AbstractVector,
                      ::PolyhedraBackwardAlgorithm)
    @assert dim(Y) == size(W, 1) == length(b) == 1
    # if Y = [l, h], then X should have two constraints:
    # ax + b <= h  <=>  ax <= h - b
    # ax + b >= l  <=>  -ax <= b - l
    l, h = low(Y, 1), high(Y, 1)
    a = vec(W)
    N = promote_type(eltype(l), eltype(b))
    if eltype(a) != N
        a = Vector{N}(a)
    end
    return HPolyhedron([HalfSpace(a, h - b[1]), HalfSpace(-a, b[1] - l)])
end

# specialization for HalfSpace
function _backward_1D(Y::HalfSpace, W::AbstractMatrix, b::AbstractVector,
                      ::PolyhedraBackwardAlgorithm)
    @assert dim(Y) == size(W, 1) == length(b) == 1
    # if Y = cx <= d (normalized: x <= d/c), then X should have one constraint:
    # ax + b <= d/c  <=>  ax <= d/c - b
    a = vec(W)
    offset = Y.b / Y.a[1] - b[1]
    N = promote_type(eltype(a), typeof(offset))
    if eltype(a) != N
        a = Vector{N}(a)
    end
    if typeof(offset) != N
        offset = N(offset)
    end
    return HalfSpace(a, offset)
end

# apply inverse affine map to one-dimensional Y
function _backward_nD(Y::LazySet, W::AbstractMatrix, b::AbstractVector,
                      ::PolyhedraBackwardAlgorithm)
    return _backward_affine_map(W, Y, b)
end

# apply inverse affine map to universe
function backward(Y::Universe{N}, W::AbstractMatrix, b::AbstractVector,
                  ::PolyhedraBackwardAlgorithm) where {N}
    @assert dim(Y) == size(W, 1) == length(b)
    return Universe{N}(size(W, 2))
end

# apply inverse affine map to union of sets
function backward(Y::UnionSetArray, W::AbstractMatrix, b::AbstractVector,
                  algo::PolyhedraBackwardAlgorithm)
    return _backward_union(Y, W, b, algo)
end

# apply inverse leaky ReLU activation function
function backward(Y::LazySet, act::LeakyReLU, algo::PolyhedraBackwardAlgorithm)
    return _backward_pwa(Y, act, algo)
end

function _backward_pwa(Y::LazySet{N}, act::ActivationFunction,
                       ::PolyhedraBackwardAlgorithm) where {N}
    @assert is_polyhedral(Y) "expected a polyhedron, got $(typeof(Y))"

    out = LazySet{N}[]
    n = dim(Y)

    for (Pj, αj) in pwa_partitioning(act, n, N)
        # inverse affine map
        αA, αb = αj
        if iszero(αA)
            # constant case
            if αb ∈ Y
                push!(out, Pj)
            end
        else
            # injective case
            R = affine_map_inverse(αA, Y, αb)
            X = intersection(Pj, R)
            push!(out, X)
        end
    end

    filter!(!isempty, out)
    return simplify_union(out; n=dim(Y), N=N)
end

# apply inverse ReLU activation function
function backward(Y::LazySet, act::ReLU, ::PolyhedraBackwardAlgorithm)
    n = dim(Y)
    if n == 1
        X = _backward_1D(Y, act)
    elseif n == 2
        X = _backward_2D(Y, act)
    else
        X = _backward_nD(Y, act)
    end
    return simplify_set(X)
end

function _backward_1D(Y::LazySet{N}, ::ReLU) where {N}
    l, h = extrema(Y, 1)
    if l <= zero(N)
        if isinf(h)
            # unbounded everywhere
            return Universe{N}(1)
        else
            # bounded from above
            return HalfSpace(N[1], N(h))
        end
    elseif isinf(h)
        # positive but unbounded from above
        return HalfSpace(N[-1], N(-l))
    else
        # positive and bounded from above, so ReLU⁻¹ = Id
        return Y
    end
end

# special case for Interval
function _backward_1D(Y::Interval{N}, ::ReLU) where {N}
    if !_leq(min(Y), zero(N))
        return Y
    else
        return HalfSpace(N[1], max(Y))
    end
end

# apply inverse ReLU activation function to 2D polyhedron
function _backward_2D(Y::LazySet{N}, ::ReLU) where {N}
    @assert is_polyhedral(Y) "expected a polyhedron, got $(typeof(Y))"

    out = LazySet{N}[]

    # intersect with nonnegative quadrant
    Q₊ = HPolyhedron([HalfSpace(N[-1, 0], zero(N)),
                      HalfSpace(N[0, -1], zero(N))])
    if Y ⊆ Q₊
        Y₊ = Y
    else
        Y₊ = intersection(Y, Q₊)
    end
    if isempty(Y₊)  # pre-image is empty if image was not nonnegative
        return EmptySet{N}(dim(Y))
    end
    if !_leq(high(Y₊, 1), zero(N)) && !_leq(high(Y₊, 2), zero(N))  # at least one positive point (assuming convexity)
        push!(out, Y₊)
    end

    # intersect with x-axis
    H₋x = HalfSpace(N[0, 1], zero(N))
    Rx = intersection(Y₊, H₋x)
    isempty_Rx = isempty(Rx)
    if !isempty_Rx
        _extend_relu_2d_xaxis!(out, Rx, H₋x, N)
    end

    # intersect with y-axis
    H₋y = HalfSpace(N[1, 0], zero(N))
    Ry = intersection(Y₊, H₋y)
    isempty_Ry = isempty(Ry)
    if !isempty_Ry
        _extend_relu_2d_yaxis!(out, Ry, H₋y, N)
    end

    # if the origin is contained, the nonpositive quadrant is part of the solution
    if !isempty_Rx && !isempty_Ry && N[0, 0] ∈ Y₊
        Tz = HPolyhedron([H₋y, H₋x])
        push!(out, Tz)
    end

    return simplify_union(out; n=dim(Y), N=N)
end

function _extend_relu_2d_xaxis!(out, R::AbstractPolyhedron, H₋, N)
    h = high(R, 1)
    if h <= zero(N)  # not relevant, case handled elsewhere
        return
    end
    l = low(R, 1)
    if isinf(h)  # upper-bound constraint redundant
        T = HPolyhedron([H₋, HalfSpace(N[-1, 0], -l)])
    else
        T = HPolyhedron([H₋, HalfSpace(N[1, 0], h), HalfSpace(N[-1, 0], -l)])
    end
    push!(out, T)
    return nothing
end

function _extend_relu_2d_yaxis!(out, R::AbstractPolyhedron, H₋, N)
    h = high(R, 2)
    if h <= zero(N)  # not relevant, case handled elsewhere
        return
    end
    l = low(R, 2)
    if isinf(h)  # upper-bound constraint redundant
        T = HPolyhedron([H₋, HalfSpace(N[0, -1], -l)])
    else
        T = HPolyhedron([H₋, HalfSpace(N[0, 1], h), HalfSpace(N[0, -1], -l)])
    end
    push!(out, T)
    return nothing
end

# apply inverse ReLU activation function to arbitrary polyhedron
#
# First compute the intersection Y₊ with the nonnegative orthant.
# Use a bitvector v, where entry 1 means "nonnegative" and entry 0 means "0".
# For instance, in 2D, v = (1, 1) stands for the positive orthant and v =(0, 1)
# stands for "x = 0". For each orthant, the corresponding preimage is the
# inverse linear map of Y₊ under Diagonal(v). Since this is a special case, it
# maps a linear constraint cx <= d to a new constraint whose normal vector is
# [c1v1, c2v2, ...], and since v is a bitvector, it acts as a filter.
function _backward_nD(Y::LazySet{N}, ::ReLU) where {N}
    @assert is_polyhedral(Y) "expected a polyhedron, got $(typeof(Y))"

    out = LazySet{N}[]
    n = dim(Y)

    # intersect with nonnegative orthant
    Q₊ = HPolyhedron([HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N)) for i in 1:n])
    Y₊ = intersection(Y, Q₊)
    if isempty(Y₊)  # pre-image is empty if image was not nonnegative
        return EmptySet{N}(dim(Y))
    end

    # find dimensions in which the set is positive (to save case distinctions)
    skip = falses(n)
    @inbounds for i in 1:n
        if !_leq(low(Y₊, i), zero(N))
            skip[i] = true
        end
    end
    fix = trues(n)  # value at non-skip indices does not matter

    for v in BitvectorIterator(skip, fix, false)
        if !any(v)
            # nonpositive orthant: case needs special treatment
            if zeros(N, n) ∈ Y₊
                Tz = HPolyhedron([HalfSpace(SingleEntryVector(i, n, one(N)), zero(N)) for i in 1:n])
                push!(out, Tz)
            end
            continue
        elseif all(v)
            # nonnegative orthant: more efficient treatment of special case
            # add if set contains at least one positive point
            if all(!_leq(high(Y₊, i), zero(N)) for i in 1:n)
                push!(out, Y₊)
            end
            # last iteration
            break
        end

        # inverse linear map
        R = linear_map_inverse(v, Y₊)

        # compute orthant corresponding to v
        constraints = HalfSpace{N,SingleEntryVector{N}}[]
        @inbounds for i in 1:n
            if v[i]
                push!(constraints, HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N)))
            else
                push!(constraints, HalfSpace(SingleEntryVector(i, n, one(N)), zero(N)))
            end
        end
        O = HPolyhedron(constraints)

        # intersect preimage of Y with orthant
        X = intersection(R, O)

        push!(out, X)
    end

    filter!(!isempty, out)
    return simplify_union(out; n=dim(Y), N=N)
end

# disambiguation
for T in (:ReLU, :LeakyReLU)
    @eval begin
        function backward(Y::UnionSetArray, act::$T, algo::PolyhedraBackwardAlgorithm)
            return _backward_union(Y, act, algo)
        end
    end
end

#################
# box algorithm #
#################

struct BoxBackwardAlgorithm <: BackwardAlgorithm end

function backward(Y::LazySet, act::ReLU, ::BoxBackwardAlgorithm)
    return _backward_box(Y, act)
end

function backward(Y::UnionSetArray, act::ReLU, ::BoxBackwardAlgorithm)
    return _backward_box(Y, act)
end

function _backward_box(Y::LazySet{N}, ::ReLU) where {N}
    n = dim(Y)

    # intersect with nonnegative orthant
    Q₊ = HPolyhedron([HalfSpace(SingleEntryVector(i, n, -one(N)), zero(N)) for i in 1:n])
    Y₊ = intersection(Y, Q₊)
    if isempty(Y₊)  # pre-image is empty if image was not nonnegative
        return EmptySet{N}(dim(Y))
    end

    constraints = Vector{HalfSpace{N,SingleEntryVector{N}}}()
    @inbounds for i in 1:n
        e₊ = SingleEntryVector(i, n, one(N))
        upper = ρ(e₊, Y₊)
        if upper < N(Inf)
            push!(constraints, HalfSpace(e₊, upper))
        end

        e₋ = SingleEntryVector(i, n, -one(N))
        lower = -ρ(e₋, Y₊)
        if !_leq(lower, zero(N))
            push!(constraints, HalfSpace(e₋, lower))
        end
    end
    if isempty(constraints)
        return Universe{N}(n)
    end
    return HPolyhedron(constraints)
end

for T in (Sigmoid, LeakyReLU)
    @eval begin
        function backward(Y::AbstractHyperrectangle, act::$T, ::BoxBackwardAlgorithm)
            l = _inverse(low(Y), act)
            h = _inverse(high(Y), act)
            return Hyperrectangle(; low=l, high=h)
        end
    end
end
