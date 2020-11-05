# Motoaki Takahashi
# November 2020
# Migration with serial correlation

# Julia 1.5.0

using Random, Distributions, LinearAlgebra

# Setting a seed by MersenneTwister
Random.seed!(123)

N = 3 # number of locations
L = 100 # number of individuals
T = 100 # the last period, in which values of all people and all locations are equalized

β = 0.95 # discount factor

# migration cost matrix
τ = ones(N, N) - Matrix(I, N, N)

# values in period T. rows are individuals, columns are locations
V_T = ones(L, N)

# generate effective labor for individuals from an (exponential of) AR(1) process
e_0 = rand(Normal(0, 1), N*L)

# set parameters for the AR(1) process
ρ_0 = 0.5
ρ_1 = 0.5

e_matrix = zeros(N*L, (T-1))
e_matrix[:,1] = ρ_0 * e_0 + ρ_1 * rand(Normal(0, 1), N*L)

for i in 2:(T-1)
    e_matrix[:,i] = ρ_0 * e_matrix[:,(i-1)] + ρ_1 * rand(Normal(0, 1), N*L)
end

# history of real wages across locations over time
real_wage = ones(N, (T-1))

# real wage (per effective labor) for individuals
real_wage_for_inds = repeat(real_wage, L, 1)

# taking exponential to get non-negative effective labor
effective_labor_matrix = exp.(e_matrix)

# then effective real income for individuals are
real_income_matrix = real_wage_for_inds .* effective_labor_matrix

# solve the individual location choice problems backward

# location of individuals (policy function)
location_choice = zeros(L, (T-1))

# value functions of individuals
value = zeros(L, (T-1))
