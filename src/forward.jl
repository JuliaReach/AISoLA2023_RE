export forward,
       forward_all,
       DefaultForwardAlgorithm,
       ConcreteForwardAlgorithm,
       LazyForwardAlgorithm,
       BoxForwardAlgorithm

abstract type ForwardAlgorithm end

########################################
# default algorithm, works for vectors #
########################################

struct DefaultForwardAlgorithm <: ForwardAlgorithm end

# propagate x through network
function forward(x,
                 net::FeedforwardNetwork,
                 algo::ForwardAlgorithm=DefaultForwardAlgorithm())
    for L in net.layers
        x = forward(x, L, algo)
    end
    return x
end

# propagate x through network and store all intermediate results
function forward_all(x,
                     net::FeedforwardNetwork,
                     algo::ForwardAlgorithm=DefaultForwardAlgorithm(),
                     results=Vector{Tuple}())
    for L in net.layers
        y, z = forward_all(x, L, algo)
        push!(results, (y, z))
        x = z
    end
    return results
end

# propagate x through layer
function forward(x, L::DenseLayerOp, algo::ForwardAlgorithm)
    y = forward(x, L.weights, L.bias, algo)  # apply affine map
    z = forward(y, L.activation, algo)  # apply activation function
    return z
end

# propagate x through layer and return all intermediate results
function forward_all(x, L::DenseLayerOp, algo::ForwardAlgorithm)
    y = forward(x, L.weights, L.bias, algo)  # apply affine map
    z = forward(y, L.activation, algo)  # apply activation function
    return y, z
end

# apply affine map to x
function forward(x, W::AbstractMatrix, b::AbstractVector, ::ForwardAlgorithm)
    return W * x + b
end

# apply identity activation function to x
function forward(x, ::Id, ::ForwardAlgorithm)
    return x
end

# apply ReLU activation function to x
function forward(x, ::ReLU, ::ForwardAlgorithm)
    return rectify(x)
end

######################
# concrete algorithm #
######################

struct ConcreteForwardAlgorithm <: ForwardAlgorithm end

# apply concrete affine map to X
function forward(X::LazySet,
                 W::AbstractMatrix,
                 b::AbstractVector,
                 ::ConcreteForwardAlgorithm)
    return affine_map(W, X, b)
end

# apply ReLU activation function to x
function forward(x, ::ReLU, ::ConcreteForwardAlgorithm)
    return concretize(rectify(x))
end

##################
# lazy algorithm #
##################

struct LazyForwardAlgorithm <: ForwardAlgorithm end

# apply lazy ReLU activation function to X
function forward(X::LazySet, ::ReLU, ::LazyForwardAlgorithm)
    return Rectification(X)
end

#################
# box algorithm #
#################

# - uses a box approximation for the ReLU activation function
#   (exploits the identity box(ReLU(X)) = ReLU(box(X)))
# - keeps identity activation functions as the identity
# - applies the affine map according to the algorithm that is passed in the
#   constructor

struct BoxForwardAlgorithm{AMA<:ForwardAlgorithm} <: ForwardAlgorithm
    affine_map_algorithm::AMA
end

# default algorithm for affine map: concrete affine map
function BoxForwardAlgorithm()
    return BoxForwardAlgorithm(LazyForwardAlgorithm())
end

# apply affine map to X according to the algorithm options
function forward(X::LazySet,
                 W::AbstractMatrix,
                 b::AbstractVector,
                 algo::BoxForwardAlgorithm)
    return box_approximation(forward(X, W, b, algo.affine_map_algorithm))
end

# apply sigmoid activation function to X
function forward(X::LazySet, act::Sigmoid, ::BoxForwardAlgorithm)
    l, h = extrema(X)
    return Hyperrectangle(; low=act(l), high=act(h))
end

# apply leaky-ReLU activation function to X
function forward(X::LazySet, act::LeakyReLU, ::BoxForwardAlgorithm)
    l, h = extrema(X)
    if !(any(isinf, l) || any(isinf, h))
        return Hyperrectangle(; low=act(l), high=act(h))
    else
        error("not implemented")
    end
end
