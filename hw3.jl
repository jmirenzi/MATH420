using LinearAlgebra
# using Plots

using Pkg
Pkg.activate("p2")
# using CairoMakie
# using GLMakie
# GLMakie.activate!()
# CairoMakie.activate!("png")
using Plots
using DelimitedFiles

files = readdir("Project_2/kn57Nodes1to57_coord/")
files = joinpath.("Project_2/kn57Nodes1to57_coord/", files)

# target = filter(x -> contains(x, "/Cloud"), files)
target = files[1]
sources = files[2:end]


"""
Process file and return matrix
"""
function process_file(fn::String)
    m = readdlm(fn, Float64, header=false)
    # n = parse(Int64, nv)
    return m
end

target = process_file(target)
sources = process_file.(sources)

function compute_center(m::Matrix{Float64})::Vector{Float64}
    n = size(m)[2]
    r = 1 / n * m * ones(n)
    return r
end

function recenter(m::Matrix{Float64})::Matrix{Float64}
    n = size(m)[2]
    m̄ = compute_center(m)
    r = m - m̄ * ones(n)'
    return r
end
function recenter(m::Matrix{Float64}, bar::Vector{Float64})::Matrix{Float64}
    n = size(m)[2]
    m̄ = bar
    r = m - m̄ * ones(n)'
    return r
end

function compute_r(x::Matrix{Float64}, y::Matrix{Float64}; center::Bool=true)::Matrix{Float64}
    if center
        r = recenter(x) * recenter(y)'
    else
        r = x * y'
    end
    return r
end

function compute_svd(x::Matrix{Float64})::NTuple{3,Matrix{Float64}}
    s = svd(x)
    r = (s.U, diagm(s.S), s.Vt)
    return r
end

"""

    Usage:
    Q, a, z = compute_qaz(sources[i], target)
"""
function compute_qaz(x::Matrix{Float64}, y::Matrix{Float64})::Tuple{Matrix{Float64},Float64,Vector{Float64}}
    x̄ = compute_center(x)
    ȳ = compute_center(y)
    X̃ = recenter(x, x̄)
    Ỹ = recenter(y, ȳ)
    u, s, vt = compute_svd(compute_r(X̃, Ỹ))
    Q = vt' * u'
    a = tr(s) / (norm(recenter(X̃)))^2
    z = x̄ - 1 / a * Q' * ȳ
    return (Q, a, z)
end

function compute_alignment_error(x::T, y::T, Q::T, a::Float64, z::Vector{Float64})::Float64 where {T<:Matrix{Float64}}
    m = a * Q * (x - z * ones(size(x)[2])') - y
    return norm(m)
end

zt(t::Float64, z::Vector{Float64})::Vector{Float64} = t * z
at(t::Float64, a::Float64)::Float64 = 1 - t + t * a
function matrix_j(Q::Matrix{Float64}, i::Int=1)::Union{Matrix{Float64},UniformScaling{Bool}}
    d = round(det(Q))
    d == 1 && return I
    if d == -1
        v = ones(size(Q)[1])
        v[i] = -1
        return diagm(v)
    else
        error("Bad determinant")
    end
end

function qt(t::Float64, Q::Matrix{Float64}, i::Int=1)::Matrix{Float64}
    J = matrix_j(Q, i)
    return J' * exp(t * log(J * Q))
end

xt(t::Float64, X::Matrix{Float64}, Q::Matrix{Float64}, a::Float64, z::Vector{Float64}; i::Int=1) = at(t, a) * qt(t, Q, i) * (X - zt(t, z) * ones(3)')
xt(t::Float64, X::Matrix{Float64}, tp::Tuple; i::Int=1) = xt(t, X, tp[1], tp[2], tp[3]; i=i)
xt(t::Float64, X::Matrix{Float64}; i::Int=1) = xt(t, X, compute_qaz(X, target); i=i)

x = sources[2]
y = target
(Q, a, z) = compute_qaz(x, y)
x1 = xt(1 / 100, x, compute_qaz(x, y))

sort(collect(eachrow(x1)))

x1

scene = Scene()
Plots.scatter(x1[:, 1], x1[:, 2], x1[:, 3])
plt = plot3d(
    1,
    xlim=(extrema(x1[:, 1])),
    ylim=(extrema(x1[:, 2])),
    zlim=(extrema(x1[:, 3])),
)
plt = plot3d(
    1,
    xlim=(extrema(x[:, 1])),
    ylim=(extrema(x[:, 2])),
    zlim=(extrema(x[:, 3])),
    legend=true,
    marker=2
)
@gif for i = 0:100
    x_ = xt(i / 100, x)
    push!(plt, x_[:, 1], x_[:, 2], x_[:, 3])
end every 1