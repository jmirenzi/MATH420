using Pkg
Pkg.activate("p2")
using DelimitedFiles
using LinearAlgebra
using Convex
using CSDP
using SCS
using SDPA
using SDPAFamily
using MATLAB    
using SDPT3 
using JuMP

function create_square_dist_matrix(R::Matrix{<:Real})::Tuple{Matrix{<:Real},Any}
    dict1 = Dict(eachrow(R[:, 1:2]) .=> R[:, 3])
    matrixR = zeros(Float64, 57, 57)
    for x in dict1
        i = trunc.(Int, x[1])
        matrixR[i[1], i[2]] = x[2]
    end
    return matrixR .^ 2, [trunc.(Int, x) for x in keys(dict1)]
end;

"""
evector(n, i, j)

Return vector e in `R^n` with +1 at `i`, -1 at `j`, and 0 else.
"""
function evector(n::Int64, i::Int64, j::Int64)::Vector{Int64}
    r = zeros(Int64, n)
    r[i] = 1
    r[j] = -1
    return r
end;

function compute_center(m::Matrix{Float64})::Vector{Float64}
    n = size(m)[2]
    r = 1 / n * m * ones(n)
    return r
end;

function recenter(m::Matrix{Float64})::Matrix{Float64}
    n = size(m)[2]
    m̄ = compute_center(m)
    r = m - m̄ * ones(n)'
    return r
end;

function recenter(m::Matrix{Float64}, bar::Vector{Float64})::Matrix{Float64}
    n = size(m)[2]
    m̄ = bar
    r = m - m̄ * ones(n)'
    return r
end;


function compute_r(x::Matrix{Float64}, y::Matrix{Float64}; center::Bool=true)::Matrix{Float64}
    if center
        r = recenter(x) * recenter(y)'
    else
        r = x * y'
    end
    return r
end;


function compute_svd(x::Matrix{Float64})::NTuple{3,Matrix{Float64}}
    s = svd(x)
    r = (s.U, diagm(s.S), s.Vt)
    return r
end;
"""

    Usage:
    Q, a, z = compute_qaz(sources[i], target)
"""
function compute_qaz(x::Matrix{Float64}, y::Matrix{Float64})::Tuple{Matrix{Float64},Float64,Vector{Float64}}
    x̄ = compute_center(x)
    ȳ = compute_center(y)
    X̃ = recenter(x, x̄)
    Ỹ = recenter(y, ȳ)
    u, s, vt = compute_svd(compute_r(X̃, Ỹ, center=false))
    Q = vt' * u'
    a = tr(s) / (norm(recenter(X̃)))^2
    z = x̄ - 1 / a * Q' * ȳ
    return (Q, a, z)
end;

function compute_alignment_error(x::T, y::T, Q::T, a::Float64, z::Vector{Float64})::Float64 where {T<:Matrix{Float64}}
    m = a * Q * (x - z * ones(size(x)[2])') - y
    return norm(m)
end;

function compute_alignment_error(x::Matrix{Float64}, y::Matrix{Float64})::Float64
    Q, a, z = compute_qaz(x, y)
    return compute_alignment_error(x, y, Q, a, z)
end;

files = readdir("Project_2/data/");
files = joinpath.("Project_2/data/", files);

observed=files[1:3];
target=files[4:6];
epsilon_list = [1,1,1];
G::Dict{Int,Matrix} = Dict([]);
# Start of Loop
# for i in 1:3
iter = 1
    observed_ = observed[iter];

    (R, (nv, m)) = readdlm(observed_, Float64, header=true);
    D, k = create_square_dist_matrix(R);
    nv = parse(Int,nv);
    m = parse(Int,m);
    # if m >= nv*(nv-1)/2 # Full data
    #     println("Full data was given?!?");

    # else
        error = epsilon_list[iter];
        G_ = Semidefinite(nv);
        problem = minimize(tr(G_));
        problem.constraints += [abs(tr(evector(nv, i[1], i[2])' * G_ * evector(nv, i[1], i[2])) - (D[i[1], i[2]])) ≤ error for i in k];

        # solve!(problem, CSDP.Optimizer)
        # solve!(problem, SCS.Optimizer)
        
        # SDPASolver(Mode=PARAMETER_UNSTABLE_BUT_FAST);
        # solve!(problem, SDPA.Optimizer)
        solve!(problem, SDPT3.Optimizer)
        # solve!(problem, SDPAFamily.Optimizer(presolve=false))
        G[iter] = G_.value;
    # end
    ev, Q = eigen(G[iter], sortby=x -> -x);
    Λ = Diagonal(ev);
    Q_1 = Q[:, 1:3];
    Λ_1 = Λ[1:3, 1:3];
    Y = Λ_1^1 / 2 * Q_1';
    min_error=0;
    for iter_2 = 1:3
        target_ = target[iter_2];
        target_data = readdlm(target_, Float64, header=false);
        align_error = compute_alignment_error(copy(transpose(Y)), target_data)
        if align_error<min_error || min_error==0
            min_error=align_error;
            matching_target = iter_2;
        end
    end
# end



