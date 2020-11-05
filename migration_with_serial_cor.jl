# Motoaki Takahashi
# November 2020
# Migration with serial correlation

# Julia 1.5.0

using Random, Distributions, LinearAlgebra, Plots

# Setting a seed by MersenneTwister
Random.seed!(123)

N = 3 # number of locations
L = 1000 # number of individuals
T = 100 # the last period, in which values of all people and all locations are equalized

β = 0.95 # discount factor

# migration cost matrix
τ = ones(N, N) - Matrix(I, N, N)

# values in period T. rows are individuals, columns are locations
V_T = (0) * ones(N*L, 1)

# generate effective labor for individuals from an (exponential of) AR(1) process
# e_0 = rand(Normal(0, 1), N*L) # random case

# set parameters for the AR(1) process
ρ_0 = 0.9
ρ_1 = 0.1

e_matrix = zeros(N*L, (T-1))
# e_matrix[:,1] = ρ_0 * e_0 + ρ_1 * rand(Normal(0, 1), N*L)
e_matrix[:,1] = repeat([1, 0, 0], L)

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

# period utility is the log of real income
period_util = log.(real_income_matrix)

# solve the individual location choice problems backward

# value functions of individuals, net of migration costs
value_net_of_mig = zeros(N*L, (T-1))

value_net_of_mig[:, (T-1)] = period_util[:, (T-1)] + β * V_T

for i in 2:(T-1)
    value_net_of_mig[:, (T-i)] = period_util[:, (T-i)] + β * value_net_of_mig[:, (T-i+1)]
end


value = ones(L, (T-1))
location_choice = zeros(L, (T-1))

for i in 1:L
    a = value_net_of_mig[((i-1)*N+1):(i*N), :]
    b = findmax(a, dims = 1)
    value[i,:] = b[1]
    # Tuple translates Cartesian indices to tuples.
    # First takes first elements from tuples
    location_choice[i,:] = first.(Tuple.(b[2]))
end

agg_population_dynamics = zeros(N, (T-1))

for i in 1:(T-1)
    agg_population_dynamics[1, i] = count(location_choice[:, i] .== 1)
    agg_population_dynamics[2, i] = count(location_choice[:, i] .== 2)
    agg_population_dynamics[3, i] = count(location_choice[:, i] .== 3)

end

# make a graph of aggregate population dynamics
x_axis = 1:(T-1)
plot(x_axis, agg_population_dynamics[1,:], title = "Aggregate Population", label = "1", legend=:topright)
plot!(x_axis, agg_population_dynamics[2,:], label = "2")
plot!(x_axis, agg_population_dynamics[3,:], label = "3")

savefig("pop_dy_1_0_0_high_ser_L1000.png")
