export bidirectional,
       SimpleBidirectionalAlgorithm,
       PolyhedraBidirectionalAlgorithm,
       BoxBidirectionalAlgorithm

abstract type BidirectionalAlgorithm end

################################
# simple algorithm combination #
################################

struct SimpleBidirectionalAlgorithm{FA<:ForwardAlgorithm,
                                    BA<:BackwardAlgorithm} <: BidirectionalAlgorithm
    fwd_algo::FA
    bwd_algo::BA
end

function fwd(algo::SimpleBidirectionalAlgorithm)
    return algo.fwd_algo
end

function bwd(algo::SimpleBidirectionalAlgorithm)
    return algo.bwd_algo
end

# convenience constructor that uses the default algorithms
function SimpleBidirectionalAlgorithm()
    return SimpleBidirectionalAlgorithm(DefaultForwardAlgorithm(),
                                        DefaultBackwardAlgorithm())
end

# convenience constructor that uses the polyhedra algorithms
function PolyhedraBidirectionalAlgorithm()
    return SimpleBidirectionalAlgorithm(ConcreteForwardAlgorithm(),
                                        PolyhedraBackwardAlgorithm())
end

# convenience constructor that uses the box algorithms
function BoxBidirectionalAlgorithm()
    return SimpleBidirectionalAlgorithm(BoxForwardAlgorithm(),
                                        BoxBackwardAlgorithm())
end

# propagate X forward and Y backward through network
function bidirectional(X, Y, net::FeedforwardNetwork,
                       algo::BidirectionalAlgorithm=SimpleBidirectionalAlgorithm();
                       get_intermediate_results::Bool=false)
    # forward propagation
    forward_results = forward_all(X, net, fwd(algo))
    if get_intermediate_results
        intermediate_results = Vector{LazySet}(undef, length(forward_results) + 1)
        @inbounds for k in eachindex(forward_results)
            forward_results[k]
            intermediate_results[k + 1] = forward_results[k][2]
        end
        fill_result = 1
    end

    bwd_algo = bwd(algo)
    first_iteration = true
    forward_result = undef

    # backward propagation with intersection
    @inbounds for k in length(net.layers):-1:1
        layer = net.layers[k]
        X_k = forward_results[k]

        # take intersection with forward set
        XY_after_activation = _fwd_bwd_intersection(algo, X_k[2], Y)
        if get_intermediate_results
            intermediate_results[k + 1] = XY_after_activation
        end

        if first_iteration
            forward_result = XY_after_activation
            first_iteration = false
        end

        # early termination (needed for compatibility)
        if XY_after_activation isa EmptySet
            Y = XY_after_activation
            if get_intermediate_results
                fill_result = k
            end
            break
        end

        # apply inverse activation function
        Y_before_activation = backward(XY_after_activation, layer.activation,
                                       bwd_algo)

        # take intersection with forward set
        XY_before_activation = _fwd_bwd_intersection(algo, X_k[1], Y_before_activation)

        # early termination (needed for compatibility)
        if XY_before_activation isa EmptySet
            Y = XY_before_activation
            if get_intermediate_results
                fill_result = k
            end
            break
        end

        # apply inverse affine map
        Y = backward(XY_before_activation, layer.weights, layer.bias, bwd_algo)
    end

    if Y isa EmptySet
        # preimage is empty
        backward_result = EmptySet{eltype(Y)}(dim(X))
    else
        # take intersection with initial set
        backward_result = _fwd_bwd_intersection(algo, Y, X)
    end

    if get_intermediate_results
        @inbounds for k in 2:fill_result
            intermediate_results[k] = EmptySet{eltype(Y)}(dim(intermediate_results[k]))
        end
        @inbounds intermediate_results[1] = backward_result

        # better output type
        intermediate_results = [e for e in intermediate_results]
    end

    return get_intermediate_results ? intermediate_results : (backward_result, forward_result)
end

function _fwd_bwd_intersection(::BidirectionalAlgorithm, X::LazySet, Y::LazySet)
    return intersection(X, Y)
end

function _fwd_bwd_intersection(::SimpleBidirectionalAlgorithm{<:BoxForwardAlgorithm,
                                                              <:BoxBackwardAlgorithm},
                               X::LazySet, Y::LazySet)
    return box_approximation(intersection(X, Y))
end
