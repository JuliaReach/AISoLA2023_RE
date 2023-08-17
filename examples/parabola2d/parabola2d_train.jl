using InverseNet
using ControllerFormats
using Plots
using Flux
using Flux: train!
using Flux.Losses: mse

include("parabola2d_initialize.jl")

current_loss = Inf
# Retry until a reasonable model is found.
THRESHOLD = 0.3
global y_predict
global model
k = 3

loss(y_predict, y_train) = mse(y_predict, y_train)

while current_loss > THRESHOLD
    global model = Chain(Dense(1 => k, relu), Dense(k => k, relu), Dense(k, 1))

    opt = Flux.setup(Adam(0.01), model)

    for epoch in 1:1000
        # train!(loss, model, data, opt)
        train!(model, data, opt) do m, x, y
            return loss(m(x), y)
        end
    end
    global y_predict = model(x_train)
    global current_loss = loss(y_predict, y_train)
    @show current_loss
end

arr2 = vec([Singleton([xi, yi]) for (xi, yi) in zip(x_train, y_predict)])
plot!(fig, UnionSetArray(arr2); color=:green, lab="model")

# Convert to ControllerFormats and serialize.
# To load it, do `deserialize(path)`.
model_cf = convert(FeedforwardNetwork, model)
path = joinpath(pkgdir(InverseNet), "examples", "parabola2d", "parabola2d.network")
write_POLAR(model_cf, path)

fig
