
using Pkg
Pkg.activate("p2")
# Pkg.instantiate()
using DelimitedFiles
using LinearAlgebra
using Plots
using LaTeXStrings
# using JuMP
using Convex
# using SCS
# using GLPK
using CSDP


(R, (nv, m)) = readdlm("Project_2/sparse/Sparse1kn57Nodes1to57_exactdist.txt", Float64, header=true)
n = parse(Int64, nv)
m = parse(Int64, m)

"""
    evector(n, i, j)

Return vector e in `R^n` with +1 at `i`, -1 at `j`, and 0 else.
"""
function evector(n::Int64, i::Int64, j::Int64)::Vector{Int64}
    r = zeros(Int64, n)
    r[i] = 1
    r[j] = -1
    return r
end
function matrixm(n::Int64, i::Int64, j::Int64)::Matrix{Int64}
    e = evector(n, i, j)
    return e * e'
end
eij = evector.(57, 1:57, 1:57)
M = eij * eij'
function create_matrix_r(R::Matrix{<:Real})::Tuple{Matrix{<:Real},Any}
    dict1 = Dict(eachrow(R[:, 1:2]) .=> R[:, 3])
    matrixR = zeros(Float64, 57, 57)
    for x in dict1
        i = trunc.(Int, x[1])
        matrixR[i[1], i[2]] = x[2]
    end
    return matrixR, [trunc.(Int, x) for x in keys(dict1)]
end
d, k1 = create_matrix_r(R)

f1(A::Matrix{<:Real}, B::Matrix{<:Real})::Real = tr(A'B)
f1(A::Matrix{<:Real}, B::Variable) = tr(A'B)

f2(A::Vector{<:Real}, B::Matrix{<:Real})::Real = (A' * B * A)
f2(A::Vector{<:Real}, B::Variable) = (A' * B * A)

G = Semidefinite(57)
# add_constraint!(G, G ≥ 0)
# add_constraint!(G, G == G')
# add_constraint!(G, G * ones(57) <= 0.001)
problem = minimize(tr(G))
er1 = 0.1
# c1 = [abs(f1(matrixm(57, trunc(Int, i[1]), trunc(Int, i[2])), G) - d[trunc(Int, i[1]), trunc(Int, i[2])]^2) ≤ er1 for i in k1]
# c1 = [abs(f2(evector(57, trunc(Int, i[1]), trunc(Int, i[2])), G) - d[trunc(Int, i[1]), trunc(Int, i[2])]^2) ≤ er1 for i in k1]
c1 = [abs(tr(evector(57, i[1], i[2])' * G * evector(57, i[1], i[2])) - (d[i[1], i[2]])^2) ≤ er1 for i in k1]
# cross(a, v::Variable) = cross()
# c1 = [abs(cross(evector(57, trunc(Int, i[1]), trunc(Int, i[2])), G) - d[trunc(Int, i[1]), trunc(Int, i[2])]^2) ≤ er1 for i in k1]
problem.constraints += c1

solve!(problem, CSDP.Optimizer)
# solve!(problem, GLPK.Optimizer)