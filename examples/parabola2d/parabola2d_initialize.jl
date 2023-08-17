using InverseNet, Plots

actual(x) = x^2 / 20
dom = Interval(-20.0, 20.0)
codom = Interval(0.0, 20.0)

# plot(dom × codom; ratio=1)
fig = plot(; ratio=1)

N = 100
s = sample(dom, N)
x_train = hcat(sort!(Float32.(getindex.(s, 1)))...)
y_train = actual.(x_train)
data = [(x_train, y_train)]

arr = vec([Singleton([xi, yi]) for (xi, yi) in zip(x_train, y_train)])
plot!(fig, UnionSetArray(arr); color=:red, lab="ground truth")
