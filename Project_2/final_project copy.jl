using Pkg
Pkg.activate("p2")
# Pkg.instantiate()
using DelimitedFiles
using LinearAlgebra
using Plots
using MATLAB
using ProgressBars
# using MATLAB    
# using SDPT3 


# cd("C:/Users/jmire/Downloads/cvx-w64/cvx/sdpt3/") do 
#     MATLAB.mat"install_sdpt3"
# end
# cd("/home/camilovelezr/cvx/") do
#     MATLAB.mat"cvx_setup"
# end
cd("/home/camilovelezr/cvx/sdpt3/") do
    MATLAB.mat"install_sdpt3"
end

function create_square_dist_matrix(R::Matrix{<:Real}, n::Int)::Tuple{Matrix{<:Real},Any}
    dict1 = Dict(eachrow(R[:, 1:2]) .=> R[:, 3])
    matrixR = zeros(Float64, n, n)
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

function qt(t::Float64, Q::Matrix{Float64}, i::Int=1)::Matrix{<:Real}
    J = matrix_j(Q, i)
    return J' * exp(t * log(J * Q))
end

function align_data(X::Matrix, Y::Matrix)
    (Q, a, z) = compute_qaz(X, Y)
    return a * Q * (X - z * ones(40)')
end

xt(t::Float64, X::Matrix{Float64}, Q::Matrix{Float64}, a::Float64, z::Vector{Float64}; i::Int=1, n::Int=40) = at(t, a) * qt(t, Q, i) * (X - zt(t, z) * ones(n)')
xt(t::Float64, X::Matrix{Float64}, tp::Tuple; i::Int=1, n::Int=40) = xt(t, X, tp[1], tp[2], tp[3]; i=i, n=n)
xt(t::Float64, X::Matrix{Float64}; i::Int=1, n::Int=40) = xt(t, X, compute_qaz(X, target); i=i, n=n)


function make_gif(x::Matrix{Float64}, y::Matrix{Float64}, title_string::String)
    x_min::Matrix{Real} = ones(100, 3)
    x_max::Matrix{Real} = ones(100, 3)
    n = size(x)[2]
    for i in 1:100
        x_ = xt(i / 100, x, compute_qaz(x, y), n=n)
        for k in 1:3
            x_min[i, k] = minimum(x_[:, k])
            x_max[i, k] = maximum(x_[:, k])
        end
    end
    up_bounds = maximum.(eachcol(x_max))
    low_bounds = minimum.(eachcol(x_min))

    @gif for i in 1:120
        x_ = xt((i > 100 ? 100 : i) / 100, x, compute_qaz(x, y), n=n)
        # Plots.scatter(x_[:, 1], x_[:, 2], x_[:, 3], xlims=(extrema(x_[:, 1])), ylims=(extrema(x_[:, 2])), zlims=(extrema(x_[:, 3])), markercolor=:blue)
        # Plots.scatter(x_[:, 1], x_[:, 2], x_[:, 3], xlims=(low_bounds[1], up_bounds[1]), ylims=(low_bounds[2], up_bounds[2]), zlims=(low_bounds[3], up_bounds[3]), markercolor=:blue)
        # Plots.scatter(x_[:, 1], x_[:, 2], x_[:, 3], markercolor=:blue)
        Plots.scatter(x_[1, :], x_[2, :], x_[3, :], markercolor=:blue)
        # Plots.scatter(y[:, 1], y[:, 2], y[:, 3], xlims=(extrema(x_[:, 1])), ylims=(extrema(x_[:, 2])), zlims=(extrema(x_[:, 3])), markercolor=:blue)
        # Plots.scatter!(y[:, 1], y[:, 2], y[:, 3], markercolor=:red)
        Plots.scatter!(y[1, :], y[2, :], y[3, :], markercolor=:red)
        title!(title_string)
    end fps = 10
end

# observed_ = observed[1];
# (R, (nv, m)) = readdlm(observed_, Float64, header=true);
# square_distance_matrix, k__ = create_square_dist_matrix(R,40);


# add the dist for both sides of the matrix
function Laplacian_eig_map(square_distance_matrix::Matrix)::Matrix
    α = 0.000001
    n = size(square_distance_matrix)[1]
    W = zeros(size(square_distance_matrix)[1], size(square_distance_matrix)[2])
    for i = 1:n, j = i:n
        # global W
        if square_distance_matrix[i, j] != 0
            W[i, j] = exp(-α * square_distance_matrix[i, j])
            W[j, i] = exp(-α * square_distance_matrix[i, j])
            #    println(W[i,j]) 
        end
    end
    D = zeros(n, n)
    for k = 1:n
        D[k, k] = sum(W[k, :])
    end
    D_root = D^(-1 / 2)
    # D_root[40,40] = 1
    laplacian_ = I(n) - D_root * W * D_root
    e_vec = eigvecs(laplacian_)
    return e_vec[2:4, :] * D_root
end


files = readdir("Project_2/data/");
files = joinpath.("Project_2/data/", files);

observed = files[1:3];
target = files[4:6];
epsilon_list = [1, 1, 1];
Y::Dict{Int,Matrix} = Dict([]);
Y_al::Dict{Int,Matrix} = Dict([]);

confusion_matrix = ones(3, 3);
matching_target = 0; # fix
# Start of Loop
for iter in ProgressBar(1:2)
    # iter = 1
    global confusion_matrix
    observed_ = observed[iter]

    (R, (nv, m)) = readdlm(observed_, Float64, header=true)

    nv = parse(Int, nv)
    @show nv
    m = parse(Int, m)
    D, k = create_square_dist_matrix(R, nv)
    @show size(D)
    SDP_bool = true
    try
        if SDP_bool
            evectors = [evector(nv, i[1], i[2]) for i in k]
            evectors = mapreduce(permutedims, vcat, evectors)
            dnums = [D[x[1], x[2]] for x in k]
            ep = 30
            MATLAB.mat"n=40;
                cvx_begin sdp
                    variable X(n,n) semidefinite;
                    minimize trace(X)
                    subject to
                        X*ones(n,1) == zeros(n,1);
                        for i=1:$(length(k))
                            ev = double($(evectors)(i,:));
                            abs(ev*X*transpose(ev) - $(dnums)(i)) <= 30
                        end
                cvx_end
                $(G) = X
                "
            error = epsilon_list[iter]
            ev, Q = eigen(G, sortby=x -> -x)
            Λ = Diagonal(ev)
            Q_1 = Q[:, 1:3]
            Λ_1 = Λ[1:3, 1:3]
            Y[iter] = (Λ_1^1 / 2 * Q_1')
        else
            # Y[iter] = copy(transpose(Laplacian_eig_map(D)));
            Y[iter] = Laplacian_eig_map(D)
        end
        min_error = 0
        for iter_2 = 1:3
            # iter_2 = 1
            global matching_target
            target_ = target[iter_2]
            target_data = copy(transpose(readdlm(target_, Float64, header=false)))
            align_error = compute_alignment_error(Y[iter], target_data)
            confusion_matrix[iter, iter_2] = align_error
            if align_error < min_error || min_error == 0
                min_error = align_error
                matching_target = iter_2
            end
        end
    catch
        rethrow()
    finally
        make_gif(Y[iter], copy(transpose(readdlm(target[matching_target]))), "Observed $(iter) to Target $(matching_target)")
    end
end
evectors = [evector(nv, i[1], i[2]) for i in k]
evectors = mapreduce(permutedims, vcat, evectors)
dnums = [D[x[1], x[2]] for x in k]
ep = 30
MATLAB.mat"n=40;
    cvx_begin sdp
        variable X(n,n) semidefinite;
        minimize trace(X)
        subject to
            X*ones(n,1) == zeros(n,1);
            for i=1:$(length(k))
                ev = double($(evectors)(i,:));
                abs(ev*X*transpose(ev) - $(dnums)(i)) <= 30
            end
    cvx_end
    $(G) = X
    "

println(confusion_matrix)


