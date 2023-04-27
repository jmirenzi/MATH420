using Pkg
Pkg.activate("p2")
using DelimitedFiles
using LinearAlgebra
using Convex
using CSDP
using SCS
using SDPA
using SDPAFamily


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
        
        SDPASolver(Mode=PARAMETER_UNSTABLE_BUT_FAST);
        solve!(problem, SDPA.Optimizer)
        # solve!(problem, SDPAFamily.Optimizer(presolve=false))
        G[iter] = G_.value;
    # end
    ev, Q = eigen(G[iter], sortby=x -> -x);
    Λ = Diagonal(ev);
    Q_1 = Q[:, 1:3];
    Λ_1 = Λ[1:3, 1:3];
    Y = Λ_1^1 / 2 * Q_1';
# end



