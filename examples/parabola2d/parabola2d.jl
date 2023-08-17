using InverseNet
using ControllerFormats
using Plots, Plots.Measures

include("parabola2d_initialize.jl")

path = joinpath(pkgdir(InverseNet), "examples", "parabola2d", "parabola2d.network")
net = read_POLAR(path)
algo = PolyhedraBackwardAlgorithm()

N = 500
seed = 0
s = sample(dom, N; seed=seed)
x_train = hcat(sort!(Float32.(getindex.(s, 1)))...)
y_train = actual.(x_train)
arr_train = vec([Singleton([xi, yi]) for (xi, yi) in zip(x_train, y_train)])
x_net = [[x] for k in 1:N for x in x_train[1, k]]  # requires different type
y_net = net.(x_net)
arr_net = vec([Singleton([xi[1], yi[1]]) for (xi, yi) in zip(x_net, y_net)])

fig2 = plot(; legendfontsize=15, tickfont=font(15), guidefontsize=25,
            xguidefont=font(30, "Times"), yguidefont=font(30, "Times"),
            bottom_margin=4mm, left_margin=0mm, right_margin=0mm,
            top_margin=2mm, size=(900, 300), leg=:top)
for k in 1:20
    local Y = Interval(k - 1, k)

    local preimage = backward(Y, net, algo)

    if preimage isa UnionSetArray
        plot!(fig2, [s × Y for s in array(preimage)];
              c=:yellow, ms=0)
    else
        plot!(fig2, preimage × Y; c=:yellow, ms=0)
    end

    # Check that the bounds are ok.
    # Inverting: actual(x) = x^2 / 20
    # lb = sqrt(low(Y, 1) * 20)
    # ub = sqrt(high(Y, 1) * 20)
    # plot!(fig2, LineSegment([-ub, 0.0], [-lb, 0.0]); lw=2.0, ls=:dash, alpha=1.0)
    # plot!(fig2, LineSegment([lb, 0.0], [ub, 0.0]); lw=2.0, ls=:dash, alpha=1.0)
end
plot!(fig2, UnionSetArray(arr_train); color=:red, ms=3, lc=:red, lab="ground truth")
plot!(fig2, UnionSetArray(arr_net); color=:blue, ms=3, marker=:diamond, lab="network")
savefig(fig2, "parabola2d.pdf")

# preimage of [100, 105]
Y = Interval(100, 105)
println("preimage of [$(low(Y, 1)), $(high(Y, 1))]:")
preimage = backward(Y, net, algo)
for (i, X) in enumerate(array(preimage))
    Xi = convert(Interval, X)
    println("set $i: [$(low(Xi, 1)), $(high(Xi, 1))]")
end

# preimage of [-1, 0]
Y = HalfSpace([1.0], 0.0)
preimage = backward(Y, net, algo)
println("preimage of ($(low(Y, 1)), $(high(Y, 1))]: $preimage")
