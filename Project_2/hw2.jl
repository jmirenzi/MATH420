
# using Pkg
# Pkg.activate("p2")
using StatsBase
# Pkg.instantiate()
using DelimitedFiles
using LinearAlgebra
using Plots
using LaTeXStrings
using Convex
# using CSDP
using SDPT3
using JLD
using ProgressBars
using CurveFit

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
        # @assert issymmetric(r)
        return r
    end
    GHW1[name] = getGram(num_vert, S, rho)
end


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

function create_square_dist_matrix(R::Matrix{<:Real})::Tuple{Matrix{<:Real},Any}
    dict1 = Dict(eachrow(R[:, 1:2]) .=> R[:, 3])
    matrixR = zeros(Float64, 57, 57)
    for x in dict1
        i = trunc.(Int, x[1])
        matrixR[i[1], i[2]] = x[2]
    end
    return matrixR .^ 2, [trunc.(Int, x) for x in keys(dict1)]
end

files = readdir("Project_2/sparse")
files = joinpath.("Project_2/sparse", files)
"""

Process file and return squared distances d with indices k
"""
function process_file(fn::String)
    (R, (nv, m)) = readdlm(fn, Float64, header=true)
    # n = parse(Int64, nv)
    m = parse(Int64, m)
    return create_square_dist_matrix(R)
end

noisy_files = last(files, 10)
noisy_files = vcat(noisy_files[2:end], noisy_files[1])

true_files = first(files, 10)
true_files = vcat(true_files[2:end], true_files[1])



process_file(files[1])
k = 1

G_est::Dict{Real,Dict{Any,Any}} = Dict([0.1 => Dict([]), 1 => Dict([])])

"""
G_est = {
    "0.1": {
        "noisy10": G
    }
}
G_est[0.1][noisy10]
"""

using MATLAB
# cd("/home/camilovelezr/cvx/") do
#     MATLAB.mat"cvx_setup"
# end
cd("/home/camilovelezr/cvx/sdpt3/") do
    MATLAB.mat"install_sdpt3"
end
# MATLAB.mat" cvx start SDP $()"

GG = []

for f in ProgressBar(vcat(true_files, noisy_files)[1:1])
    d, k1 = process_file(f)
    for er1 in [0.1]
        G = Semidefinite(57)
        problem = minimize(tr(G))
        c1 = [abs(tr(evector(57, i[1], i[2])' * G * evector(57, i[1], i[2])) - (d[i[1], i[2]])) ≤ er1 for i in k1]
        problem.constraints += c1

        solve!(problem, SDPT3.Optimizer)
        push!(GG, G)
        G_est[er1][f] = G.value
    end
end
d1, k1 = process_file(noisy_files[10])
G1 = Semidefinite(57)
problem = minimize(tr(G1))
c1 = [abs(tr(evector(57, i[1], i[2])' * G1 * evector(57, i[1], i[2])) - (d1[i[1], i[2]])) ≤ 1 for i in k1]
problem.constraints += c1

solve!(problem, SDPT3.Optimizer)

k1
d1, k1 = process_file(noisy_files[10])
evectors = [evector(57, i[1], i[2]) for i in k1]
# evectors = hcat(evectors)
evectors = mapreduce(permutedims, vcat, evectors)

dnums = [d1[x[1], x[2]] for x in k1]
dnums = hcat(dnums)

# MATLAB.mat"n=57;
#        cvx_begin sdp
#            variable X(n,n) semidefinite;
#            minimize trace(X)
#            subject to
#              X*ones(n,1) == zeros(n,1);
#              for i=1:$(length(k1))
#                 k_ = $(k1)(i)
#                 disp(k_)
#                 ev = $(evectors)(i)
#                 abs(trace(transpose(ev)*X*ev) - $(d1)(k_(1), k_(2))) <= 0.1
#              end
#        cvx_end
#        $Xm = X
#        "

# dnums = [d1[x[1], x[2]] for x in k1]

mat"n=57; ev = $(evectors); disp(ev(1,54))"

MATLAB.mat"n=57;
       cvx_begin sdp
           variable X(n,n) semidefinite;
           minimize trace(X)
           subject to
             X*ones(n,1) == zeros(n,1);
             for i=1:$(length(k1))
                ev = double($(evectors)(i,:))
                abs(trace(ev*X*transpose(ev)) - $(dnums)(i)) <= 1
             end
       cvx_end
       $(Xm) = X
       "


@save "./G_MATLAB_TEST" G_est

# G_est = load("./G_est_dict")["G_est"]

Error::Dict{Real,Vector} = Dict([0.1 => ones(10), 1 => ones(10)])
Error_Noisy::Dict{Real,Vector} = Dict([0.1 => ones(10), 1 => ones(10)])
for er1 = [0.1, 1]
    k = 1
    for f in ProgressBar(true_files)
        Error[er1][k] = norm(GHW1["exactdist"] - G_est[er1][f])
        k += 1
    end
    k = 1
    for f in noisy_files
        Error_Noisy[er1][k] = norm(GHW1["dist"] - G_est[er1][f])
        k += 1
    end
end

display(plot(range(1, 10, 10), [Error[0.1] Error_Noisy[0.1]], title=L"\epsilon=.1", label=["True Data" "Noisy Data"], xlabel="Increments of m", ylabel="Error"))
display(plot(range(1, 10, 10), [Error[1] Error_Noisy[1]], title=L"\epsilon=1", label=["True Data" "Noisy Data"], xlabel="Increments of m", ylabel="Error"))

m(x) = (x / 10) * 57 * (56) / 2
for ep = [0.1 1]
    b_ = (log(Error[ep][end]) - log(Error[ep][1])) / (m(10) - m(1))
    a_ = Error[ep][1] / exp(b_ * m(1))
    pf2(x) = a_ * exp(b_ * x)
    display(plot(1:10, [(Error[ep]) (pf2.(m.(range(1, 10))))], title=L"\epsilon=%$(ep) \quad Error\sim e^{-mb}", label=["Error" "Error Estimation"]))

    c_ = exp((log(m(10), E[ep][10]) - log(m(1), E[ep][1])) / (1 / log(m(10)) - 1 / log(m(1))))
    d_ = -1 * log(m(1), Error[ep][1] / c_)
    pf3(x) = c_ * x^(-1 * d_)
    display(plot(1:10, [(Error[ep]) (pf3.(m.(1:10)))], title=L"\epsilon=%$(ep) \quad Error\sim \frac{1}{m^{\alpha}}", label=["Error" "Error Estimation"]))
end


G_est[1.0][files[end]]

MATLAB.mat"m=20; n=10;
       E1 = randn(n,n); d1 = randn(n,1);
       E2 = randn(n,n); d2 = randn(n,1);
       epsx = 1e-1;
       cvx_begin sdp
           variable X(n,n) semidefinite;
           minimize trace(X)
           subject to
             X*ones(n,1) == zeros(n,1);
             abs(trace(E1*X)-d1)<=epsx;
             abs(trace(E2*X)-d2)<=epsx;
       cvx_end
       $Xm = X
       "

