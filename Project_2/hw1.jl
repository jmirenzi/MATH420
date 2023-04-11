using DelimitedFiles
using LinearAlgebra

(R, num_vert_) = readdlm("Project_2/kn57Nodes1to57_dist.txt", Float64, header=true);
num_vert = parse(Float64, num_vert_[1]);
# S = zeros(size(R))
# rho = 0;
# for i = 1:57, j = 1:57
#     global S, R, rho
#     S[i, j] = R[i, j]^2
#     rho += S[i, j]
# end
S = R .^ 2
rho = norm(S, 1)
rho \= (2 * num_vert);
one_col = ones(57, 1);
v = (S - rho * I) * one_col / num_vert;


