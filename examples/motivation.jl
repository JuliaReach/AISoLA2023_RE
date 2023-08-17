using InverseNet, ControllerFormats, Plots, Plots.Measures
import Polyhedra, CDDLib
seed = 100

#############
# Example 1 #
#############

W = [-0.46 0.32;]
b = [2.0]
Y = Interval(2.0, 3.0)
algo = PolyhedraBackwardAlgorithm()

X = backward(Y, W, b, algo)

#############
# Example 3 #
#############

Y = X

X = backward(Y, ReLU(), algo)

@assert length(X) == 3  # one set is empty
X₁ = X[1]
X₂ = X[2]
X₃ = ∅(2)
X₄ = X[3]

#############
# Example 4 #
#############

function plot_raw()
    return plot(; legendfontsize=15,
                tickfont=font(25),
                guidefontsize=25,
                xguidefont=font(30, "Times"),
                yguidefont=font(30, "Times"),
                bottom_margin=0mm,
                left_margin=0mm,
                right_margin=0mm,
                top_margin=0mm,
                size=(900, 600))
end

function plot_samples!(x1, x2)
    plot!(Singleton(x2[1]); lab="y1 ≤ y2", color=:blue, alpha=1, ms=6)
    plot!(Singleton(x1[1]); lab="y1 > y2", color=:red, alpha=1, markersize=6)
    plot!(Singleton.(x2[2:end]); lab="", color=:blue, alpha=1, ms=6)
    return plot!(Singleton.(x1[2:end]); lab="", color=:red, alpha=1, markersize=6)
end

# define a neural network

W1 = [0.30 0.53
      0.77 0.42]
b1 = [0.43
      -0.42]
l1 = DenseLayerOp(W1, b1, ReLU())
W2 = [0.17 -0.07
      0.71 -0.06]
b2 = [-0.01
      0.49]
l2 = DenseLayerOp(W2, b2, ReLU())
W3 = [0.35 0.17
      -0.04 0.08]
b3 = [0.03
      0.17]
l3 = DenseLayerOp(W3, b3, Id())
layers = [l1, l2, l3]
net = FeedforwardNetwork(layers)

# plot samples

dom = BallInf([0.5, 0.5], 0.5)
x = sample(dom, 300; seed=seed)
y = net.(x)
x1 = [x[i] for i in eachindex(x) if y[i][1] >= y[i][2]]
x2 = [x[i] for i in eachindex(x) if y[i][1] < y[i][2]]
y1 = [yi for yi in y if yi[1] >= yi[2]]
y2 = [yi for yi in y if yi[1] < yi[2]]

plot_raw()
plot!(; top_margin=3mm, right_margin=5mm)
plot_samples!(x1, x2)
xlims!(low(dom, 1), high(dom, 1))
ylims!(low(dom, 2), high(dom, 2))
savefig("original_samples.pdf")

# forward image under domain

Y = forward(dom, net, ConcreteForwardAlgorithm())

plot_raw()
lab = "N([0, 1]²)"
plot!(Y; c=:blue, lab=lab, lw=2)
plot!([0.0, 1.0], [0.0, 1.0]; c=:black, lw=3, la=1, label="y1 = y2")
B = box_approximation(Y)
xlims!(low(B, 1), high(B, 1))
ylims!(low(B, 2), high(B, 2))
plot_samples!(y1, y2)
savefig("forward_samples.pdf")

# backward image under subset y2 >= y1

algo = PolyhedraBackwardAlgorithm()
c = color_list(:default)

Y1 = HalfSpace([1.0, -1.0], 0.0)

Y2 = backward(Y1, l3, algo)
plot_raw()
plot!(; top_margin=2mm)
xlims!(-0.3, 0.5)
ylims!(-1, 2)
plot!(Y2; c=c[1])
plot!([0.0, 0.0], [0.0, 2.0]; c=:black, ls=:dash, lw=2, lab="")
plot!([0.0, 0.5], [0.0, 0.0]; c=:black, ls=:dash, lw=2, lab="")
savefig("backward_l3.pdf")

Y3 = backward(Y2, ReLU(), algo)
plot_raw()
plot!(; top_margin=2mm)
xlims!(-0.3, 0.5)
ylims!(-1, 2)
plot!(Y3[1]; c=c[1])
plot!(Y3[3]; c=c[2])
plot!(Y3[4]; c=c[3])
plot!(Y3[2]; c=c[4])
savefig("backward_l3relu.pdf")

Y4 = backward(Y3, W2, b2, algo)
plot_raw()
plot!(; right_margin=1mm)
xlims!(-2, 3)
ylims!(-10, 12)
plot!(Y4[1]; c=c[1])
plot!(Y4[3]; c=c[2])
plot!(Y4[4]; c=c[3])
plot!(Y4[2]; c=c[4])
plot!([0.0, 0.0], [0.0, 12.0]; c=:black, ls=:dash, lw=2, lab="")
plot!([0.0, 3.0], [0.0, 0.0]; c=:black, ls=:dash, lw=2, lab="")
savefig("backward_l2.pdf")
# color c[4] gets removed

Y5 = backward(Y4, ReLU(), algo)
plot_raw()
plot!(; right_margin=1mm)
xlims!(-2, 3)
ylims!(-10, 12)
plot!(Y5[1]; c=c[1])
plot!(Y5[3]; c=c[2])
plot!(Y5[7]; c=c[3])
plot!(Y5[4]; c=c[5])
plot!(Y5[6]; c=c[6])
plot!(Y5[5]; c=c[7])
plot!(Y5[8]; c=c[8])
plot!(Y5[2]; c=c[9])
savefig("backward_l2relu.pdf")

Y6 = backward(Y5, W1, b1, algo)
plot_raw()
plot!(; top_margin=2mm)
xlims!(-2, 22)
ylims!(-12, 5)
plot!(Y6[1]; c=c[1])
plot!(Y6[3]; c=c[2])
plot!(Y6[7]; c=c[3])
plot!(Y6[4]; c=c[5])
plot!(Y6[6]; c=c[6])
plot!(Y6[5]; c=c[7])
plot!(Y6[8]; c=c[8])
plot!(Y6[2]; c=c[9])
plot!(dom; alpha=0, lc=:red, lw=5, la=1, ls=:dot)
savefig("backward_l1.pdf")

plot_raw()
plot!(; top_margin=3mm, right_margin=5mm)
plot!(Y6[1]; c=c[1])
plot!(Y6[2]; c=c[9])
xlims!(low(dom, 1), high(dom, 1))
ylims!(low(dom, 2), high(dom, 2))
plot_samples!(x1, x2)
savefig("backward_samples.pdf")
