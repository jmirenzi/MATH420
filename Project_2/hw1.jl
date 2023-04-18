using DelimitedFiles
using LinearAlgebra
using Plots
using LaTeXStrings

export GHW1

GHW1 = Dict{String,Any}("exactdist" => 0, "dist" => 0)
for name = ("dist", "exactdist")
    (R, num_vert_) = readdlm("Project_2/kn57Nodes1to57_" * name * ".txt", Float64, header=true)
    num_vert = parse(Float64, num_vert_[1])

    S = R .^ 2

    one_col = ones(57, 1)
    rho = 1 / (2num_vert) * one_col' * S * one_col
    rho = rho[1]

    v = ((S - rho * I(57)) * one_col) / num_vert
    function getGram(n::Real, S::Matrix, rho::Real)::Matrix
        r = 1 / 2n * (S - rho * I) * one_col * one_col' + 1 / 2n * one_col * one_col' * (S - rho * I) - 1 / 2 * S
        @assert issymmetric(r)
        return r
    end
    GHW1[name] = getGram(num_vert, S, rho)
end
#     ev, Q = eigen(G, sortby=x -> -x)
#     index = findfirst(x -> x < 0, ev)
#     ev[index-1:end] .= 0
#     display(scatter(ev, labels=nothing, title=L"Eigenvalues \quad for \quad G for \quad %$(name)", xaxis=L"Edges", yaxis=L"Value"))
#     print("The 10 largest eigenvalues are $(first(ev, 10))")
#     Λ = Diagonal(ev)
#     lambda = Λ
#     Q[:, 1:2]

#     Y::Dict{Int,Matrix} = Dict([])
#     R_hat::Dict{Int,Matrix} = Dict([])
#     R_norm::Dict{Int,Real} = Dict([])
#     ϵ::Dict{Int,Real} = Dict([])
#     σ::Dict{Int,Real} = Dict([])

#     println("For data " * name)
#     for d = (2, 3)
#         println("d = $(d)")
#         Q_1 = Q[:, 1:d]
#         Λ_1 = Λ[1:d, 1:d]
#         Y[d] = Λ_1^1 / 2 * Q_1'
#         R_hat[d] = zeros(57, 57)
#         for i = 1:57, j = 1:57
#             R_hat[d][i, j] = norm(Y[d][1:d, j] - Y[d][1:d, i])
#         end
#         R_norm[d] = norm(R - R_hat[d])
#         println("Norm = $(R_norm[d])")
#         ϵ[d] = norm(G - Y[d]'Y[d])
#         println("ϵ = $(ϵ[d])")
#         σ[d] = sqrt(sum(ev[d+1:end] .^ 2))
#         println("σ = $(σ[d])")
#         println("Difference = $(ϵ[d] - σ[d])")
#     end
#     display(scatter(Y[2][1, :], Y[2][2, :], title="Two dimensional Embedding for " * name))
#     display(scatter3d(Y[3][1, :], Y[3][2, :], Y[3][3, :], title="Three dimensional Embedding for " * name))
#     # scatter3d(Y[3][1,:], Y[3][2,:], Y[3][3,:], camera=[0,0,0])
# end