using Test, InverseNet
using ControllerFormats

include("example_networks.jl")

@time @testset "Forward" begin
    include("forward.jl")
end
@time @testset "Backward" begin
    include("backward.jl")
end
@time @testset "Bidirectional" begin
    include("bidirectional.jl")
end

@test isempty(detect_ambiguities(InverseNet))
