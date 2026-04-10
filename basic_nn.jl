using Random
using Distributions: Binomial


# =============================================================================================
# 1) Neural-network strategy representation for the evolutionary simulation
# =============================================================================================
module NeuralNetworkRepresentation
using Random

# ---------------------------------------------------------------------------------------------
# Struct to store the fully-connected feedforward network
# ---------------------------------------------------------------------------------------------
"""
Fields:
- layers::Vector{Int}, a vector of integers specifying how many nodes for each layer
- W::Vector{Matrix{Float64}}, a vector of weight matrices (one matrix represents weights for layer i to i+1)
- b::Union{Nothing, Vector{Vector{Float64}}}, bias term: either null or a vector of floats
- hidden_act::Function, activation function for hidden layers
- out_act::Function, activation function for the output layer
"""
struct MLP
    layers::Vector{Int}
    W::Vector{Matrix{Float64}}
    b::Union{Nothing, Vector{Vector{Float64}}}
    hidden_act::Function
    out_act::Function
end

# ---------------------------------------------------------------------------------------------
# Activation functions
# ---------------------------------------------------------------------------------------------
identity_act(x) = x
sigmoid(x) = 1/(1+exp(-x))
relu(x) = max(0, x)

# ---------------------------------------------------------------------------------------------
# Initialise the network
# ---------------------------------------------------------------------------------------------
"""
    init_mlp(layer; use_bias=true, hidden_act=tanh, out_act=identity, rng=Random.default_rng())

Creates a feed-forward Multi Layered Perceptron with random parameters.

Arguments
---------
- layers: vector of integers specifying the width of each layer including input and output
- use_bias(Bool), whether to use biases or not
- hidden_act(Function), activation function for hidden layers
- out_act(Function), activation function for output layer
- rng(AbstractRNG), random number generator

Returns
--------
An "MLP" object with weights and (optionally) biases drawn i.i.d. from Uniform (0,1).
Output layer forced to have exactly 1 node.
"""

function init_mlp(layers::AbstractVector{<:Integer};
                    use_bias::Bool = true,
                    hidden_act::Function = tanh,
                    out_act::Function = sigmoid,
                    rng::AbstractRNG = Random.default_rng()
                )
    @assert length(layers) >= 2 "Need at least two layers"

    # Iterate over "layers", convert each element to Int, then store them in a new Vector{Int}
    layers_vec = collect(Int, layers)

    # Force the last layer to have exactly one node
    if layers_vec[end] != 1
        println("Forcing last layer to have only one node")
        layers_vec[end] = 1
    end

    # Compute the number of matrices needed to store edges values
    L = length(layers_vec) - 1

    # Create the weight matrices
    # W is a vector of L matrices (yet undefined)
    W = Vector{Matrix{Float64}}(undef, L)
    # If use_bias is True, b is a vector of L vectors (yet undefined), otherwise nothing
    b = use_bias ? Vector{Vector{Float64}}(undef, L) : nothing
    
    # Initialise the weight matrices. Loop over the number of matrices
    for l in 1:L
        # Retrieve the number of in_dimension and out_dimensions
        in_dim = layers_vec[l]
        out_dim = layers_vec[l+1]

        # Fill a matrix of dimension (out_dim x in_dim) with entries drawn at random from a Uniform(0,1)
        W[l] = rand(rng, out_dim, in_dim)

        # If use_bias == true, draw also the vector of biases
        if use_bias
            b[l] = rand(rng, out_dim)
        end
    end

    return MLP(layers_vec, W, b, hidden_act, out_act)
end

# ---------------------------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------------------------
"""
    forward(net, x)

Compute the network output for a given input vector (x). Returns a scalar
"""

function forward(net::MLP, x::AbstractVector{<:Real})
    # Convert input to Float64 vector
    a = Float64.(x)

    @assert length(a) == net.layers[1] "Input has dimension $(length(a)), expected $(net.layers[1])."

    # Retrieve 
    L = length(net.W)

    for l in 1:L
        # Compute pre-activation z = W*a (+b if present)
        z = net.W[l]*a
        if net.b !== nothing
            z .+= net.b[l]
        end

        # Apply activation
        if l<L
            a = net.hidden_act.(z)
        else
            a = net.out_act.(z)
        end
    end

    return a[1]
end

end

# =============================================================================================
# 2) Mutation
# =============================================================================================
module Mutation
using ..NeuralNetworkRepresentation: MLP, forward
using Random

"""
    deepcopy_mlp(net)

Makes a deep copy of the network, duplicating weights and biases
"""
function deepcopy_mlp(net::MLP)
    Wc = [copy(W) for W in net.W]
    bc = net.b === nothing ? nothing : [copy(b) for b in net.b]
    return MLP(copy(net.layers), Wc, bc, net.hidden_act, net.out_act)
end

# 3.1 Gaussian shock mutation
"""
    mutate_weights_gaussian!(net, sigma=0.05, rng=Random.default_rng())

Adds iid Normal(0, sigma) noise to all weights (and bias if present). Mutates the network in place!
"""

function mutate_weights_gaussian!(net::MLP;
                                sigma::Float64 = 0.05,
                                rng::AbstractRNG = Random.default_rng())
    # Loop over each matrix in the vector W, and add a gaussian iid shock to each weight
    for l in 1:length(net.W)
        net.W[l] .+= sigma .* randn(rng, size(net.W[l])...)
        if net.b !== nothing
            net.b[l] .+= sigma .* randn(rng, length(net.b[l]))
        end
    end
    return net        
end


# 3.2 Hidden node addition
"""
    add_hidden_node(net; rng=Random.default_rng())
"""

function add_hidden_node(net::MLP; 
                        rng::AbstractRNG = Random.default_rng())
    @assert length(net.layers)==3 "add_hidden_note currently supports only 3-layered networks"
    @assert net.layers[1] == 1 && net.layers[end] == 1 "Either the first or the last layer are not 1-node"

    newnet = deepcopy_mlp(net)

    # Adjust the "layers" vector
    H = newnet.layers[2]
    newH = H+1
    newnet.layers[2] = newH

    # Adjust the weight matrices. First, add a row to the first matrix
    W1 = newnet.W[1]
    newnet.W[1] = vcat(W1, rand(rng, 1, size(W1,2)))

    # Then add a column to the second matrix
    W2 = newnet.W[2]
    newnet.W[2] = hcat(W2, rand(rng, size(W2,1), 1))
    
    # Expand biases if present
    if newnet.b !== nothing
        newnet.b[1] = vcat(newnet.b[1], rand(rng,1))
    end
    
    return newnet
end

# 3.3 Hidden node delection
"""
    delete_hidden_node(net; rng=Random.default_rng())

Deletes one node from the hidden layer. If the size of the hidden layer is 1, returns the original net
Returns a new network after making a copy of the input one.
"""
function delete_hidden_node(net::MLP; 
                            rng::AbstractRNG = Random.default_rng())
    @assert length(net.layers)==3 "delete_hidden_node currently supports only 3-layered networks"
    @assert net.layers[1] == 1 && net.layers[end] == 1 "Either the first or the last layer are not 1-node"
    
    H = net.layers[2]
    if H <= 1
        return deepcopy_mlp(net)
    end

    newnet = deepcopy_mlp(net)
    drop = rand(rng, 1:H)

    # Adjust the "layers" vector
    newnet.layers[2] = H-1

    # Adjust the weight matrices. Remove the row "drop" from W1 andf the column "drop from W2
    newnet.W[1] = newnet.W[1][setdiff(1:H, drop), :]
    newnet.W[2] = newnet.W[2][:, setdiff(1:H, drop)]
    
    # Remove the element "drop" from the bias vector if present
    if newnet.b !== nothing
        newnet.b[1] = newnet.b[1][setdiff(1:H, drop)]
    end
    
    return newnet
end


# 3.* Mutation
"""
    mutate_nn(net, rn; p_add=0.2, p_del=0.2, sigma=0.05, rng=Random.default_rng())

We specify mutation as follows:
1. Node addition
    * If this mutation is selected, select randomly to which layer the node is going to be attached
    * The new node is initialised with some random weights
2. Node deletion
    * If this mutation is selected, select randomly to which layer the node is going to be attached
    * Randomly delete one node in the selected layer
3. Weight change: iid gaussian weight shock

Inputs
-------
- net
- rn: a Uniform(0,1) draw used to choose the mutation type

Outputs
-------
- A new network
"""
function mutate_nn(net::MLP,
                rn::Float64;
                p_add::Float64 = 0.2,
                p_del::Float64 = 0.2,
                sigma::Float64 = 0.05,
                rng::AbstractRNG = Random.default_rng())
    @assert 0.0 <= rn <= 1.0
    @assert p_add >=0 && p_del >=0 && (p_add + p_del) <= 1.0

    if rn < p_add
        return add_hidden_node(net; rng=rng)
    elseif rn < p_add+p_del
        return delete_hidden_node(net; rng=rng)
    else
        newnet = deepcopy_mlp(net)
        mutate_weights_gaussian!(newnet; sigma=sigma, rng=rng)
        return newnet
    end
end

end

# =============================================================================================
# 3) Selection
# =============================================================================================
module Selection
using ..Mutation: deepcopy_mlp
using Random

"""
Performs a selection step.

Input:
- payoffs: vector of payoffs for each individual
- n
- RI
- Pairs


Output:
- newinds: indices of the dying individuals
- repstr: indices of the reproducing individuals 
"""

function Moran_selection(payoffs,N, RI,Pairs) #RI the number of replacing individuals
    # returns the vector of indices that are new, and a reference to what strategy each of these new individuals play, in reference to the old population
    payoffs = payoffs/sum(payoffs);
    # Transform payoffs into a cdf
    for i in 2:N
        payoffs[i] = payoffs[i]+ payoffs[i-1];  
    end 
    # Vector of RI independent uniform (0,1) draws
    nr  = rand(RI);#new randowm draw

    # Create a vector of zeros of length RI
    repstr = zeros(Int64,RI);
    
    # For each independent draw, find the smallest index x such that cdf(x) exceeds it
    for i in 1:RI
        repstr[i] = findfirst( x -> payoffs[x] > nr[i], 1:N); ##
    end
    # Pick RI individuals to replace by selecting RI/2 pairs at random
    # Shuffle N/2 pair indices, then take the first RI/2, take those columns from Pairs, then vectorize that matrix
    newinds = vec(Pairs[:, shuffle(1:div(N,2))[1:div(RI,2)]]); #the ones that died for this
    return newinds, repstr
end 

"""
Performs the selection: updates the population, reset actions, fixes the pairing structure
Inputs:
- payoffs
- Pairs
- Population
- state_v
- N
- RI

Outputs:
- Pairs
- Population
- state_v
"""
function pop_selection(payoffs, Pairs, Population, past_actions, N, RI)
    ## new_inds are the indivuals that are being replaced
    ## repstr are the reproducing strategies --> so in the population object the new_inds will have their strategy updated to the strategies in the repstr object
    new_inds, repstr  = Moran_selection(payoffs, N, RI, Pairs); #the new individuals
    
    # Take the Population vector and replace the neural networks with the reproducing ones
    for k in eachindex(new_inds)
        Population[new_inds[k]] = deepcopy_mlp(Population[repstr[k]])
        past_actions[new_inds[k]] = past_actions[repstr[k]]
    end

    # Create two booleans a and b with length = number of pairs where a tells if first member of Pairs is among "new indices"
    # i.e. indices that are to be replaced
    a = in(new_inds).(Pairs[1,:]);
    b = in(new_inds).(Pairs[2,:]);
    
    # Sum the two boolean vectors (0 only if both members of the pair are not to be replaced)
    changed = (a.+b);

    # Keep only pairs for which changed is 0
    Pairs = Pairs[:,(changed.==0)]
    
    return Pairs, Population, past_actions 
end

end 

# =============================================================================================
# 5) Matching
# =============================================================================================
module Matching
using Random

"""
Randomly breaks up some pairs.
Input:
- Pairs: matrix 2*(N/2) containing the pairs
- beta: Beta is taken as the per-individual continuation probability, so beta**2 is the probability that both individuals in a pair survive

Output:
- Pairs without columns
"""
function pop_breaks(Pairs,beta)
    # Take the number of pairs and draw one uniform random number per pair
    r = rand(size(Pairs,2));
    # In the matrix Pairs, take the columns that have r lower than Beta**2
    Pairs = Pairs[:, (r.<(beta^2))]; # the events are i.i.d.
    return Pairs
end

## perform rematching, and set states of rematched ones to one
function pop_matching(Pairs,otN, past_actions) 
    # Compare otN (1:N) with individuals in Pairs, then return the difference
    singles = setdiff(otN,vec(Pairs));
    @assert iseven(length(singles))

    # Randomly permute the singles
    singles = shuffle(singles);

    # Horizontally concatenate Pairs with a reshaped version of singles (2 rows, as many columns as needed)
    Pairs = hcat(Pairs,reshape(singles, 2,:));
    return Pairs, past_actions
end

"""
pair_types =function(Pairs,actions)
    # how many pairs play CC, CD or DD
    a_i(i) = actions[i];
    t = zeros(3);
    t_pre=a_i.(Pairs);
    t_pre = sum(t_pre,dims=1);
    t[1] = sum(t_pre.==2);
    t[2] = sum(t_pre.==3);
    t[3] = sum(t_pre.==4);
    return t
end 
"""

end

# =============================================================================================
# 6) Simulations
# =============================================================================================
module Simulations
using ..NeuralNetworkRepresentation: MLP, forward, init_mlp
using ..Mutation: deepcopy_mlp, mutate_nn
using ..Selection: pop_selection
using ..Matching: pop_breaks, pop_matching
using Distributions: Binomial
using Statistics: mean, std
using Random

# ---------------------------------------------------------------------------------------------
# Struct to store relevant information about payoffs
# ---------------------------------------------------------------------------------------------
struct SimulationHistory
    avg_coop::Vector{Float64}
    sd_coop::Vector{Float64}
    avg_hidden_nodes::Vector{Float64}
    avg_payoff::Vector{Float64}
    mean_response_to_C::Vector{Float64}
    mean_response_to_D::Vector{Float64}
    hidden_nodes_dist::Vector{Vector{Int}}
end

# ---------------------------------------------------------------------------------------------
# Action choice for the population of NNs
# ---------------------------------------------------------------------------------------------
"""
    nn_pop_actions(population, past_action; epsilon=0.0, rng=Random.default_rng())

Computes current contributions a_i ∈ [0,1] for each individual i.

Inputs
------
- population::Vector{MLP}
    population[i] is individual i's network.
- past_action::Vector{Float64}
    past_action[i] is the scalar input fed to i's network (for now: i's own action last period).
- epsilon::Float64
    Implementation error probability.
    Continuous analogue: with prob epsilon, we replace a_i by (1 - a_i).
- rng::AbstractRNG

Returns
-------
- actions::Vector{Float64} with values in [0,1].
"""
function nn_pop_actions(population::Vector{MLP},
                        past_action::Vector{Float64},
                        Pairs::Matrix{Int},
                        epsilon::Float64 = 0.0,
                        rng::AbstractRNG = Random.default_rng())

    # Retrieve the number of individuals
    N = length(population)
    @assert length(past_action) == N

    # Create the array to store each individual's action
    actions = Vector{Float64}(undef, N)

    # Loop over the population and compute the forward pass given the network and the past action
    @inbounds for i in 1:N
        # Retrieve the past action of the opponent: (find the item in Pairs, take the column index, index Pairs with the given
        # index to take the column, take the setdifference between the column and the player)
        # TODO: can crush if matching is wrong
        col = findfirst(x -> x == i, Pairs)[2]
        pair = Pairs[:, col]
        j = pair[1] == i ? pair[2] : pair[1]
        past_action_opponent = [past_action[j]]

        # TODO: extend to more complicated networks
        # Forward pass on the network with past action of the opponent as input
        a = forward(population[i], past_action_opponent)
        
        # Clamp for safety
        a = clamp(a, 0.0, 1.0)

        # Tremble with probability epsilon, invert the contribution
        if epsilon > 0.0 && rand(rng) < epsilon
            a = 1.0 - a
        end

        actions[i] = a
    end
    return actions
end

# ---------------------------------------------------------------------------------------------
# Payoffs for continuous contributions (still using a PD matrix)
# ---------------------------------------------------------------------------------------------
"""
    nn_pop_payoffs(Pairs, actions, payoff2b2, w)

The output of the nn is in [0, 1]. I interpret this as the probability of playing C.
"""
function nn_pop_payoffs(Pairs::Matrix{Int},
                        actions::Vector{Float64},
                        N::Int,
                        payoff2b2::Matrix{Int},
                        w::Float64)

    payoffs = zeros(Float64,N)

    # Extract matrix entries
    R = Float64(payoff2b2[1,1])
    S = Float64(payoff2b2[1,2])
    T = Float64(payoff2b2[2,1])
    P = Float64(payoff2b2[2,2])
    
    # Interpreting a_i as probability of playing "cooperate", loop over each pair and compute the realised payoffs
    # Loop over the colums of Pairs
    @inbounds for k in 1:size(Pairs, 2)
        # Retrieve the indices of the two members of the pair
        i = Pairs[1,k]
        j = Pairs[2,k]

        # Retrieve from actions the probability that each member cooperates
        c_i = actions[i]
        c_j = actions[j]

        # Compute the realised payoff
        payoffs[i] = c_i*c_j*R + c_i*(1-c_j)*S + (1-c_i)*c_j*T + (1-c_i)*(1-c_j)*P
        payoffs[j] = c_i*c_j*R + c_j*(1-c_i)*S + (1-c_j)*c_i*T + (1-c_i)*(1-c_j)*P
    end

    # Compute the final payoffs as a convex combination of 1 and the payoffs from the game, weighted by w
    payoffs = (1-w) .* ones(N) .+ w .* payoffs
    return payoffs
end

# ---------------------------------------------------------------------------------------------
# Memory update
# ---------------------------------------------------------------------------------------------
"""
    nn_update_memory!(past_action, actions)

In the current version: each individual's next-period input is simply their own last action.
"""
function nn_update_memory!(past_action::Vector{Float64},
                            actions::Vector{Float64})
    @assert length(past_action) == length(actions)
    past_action .= actions
    return nothing
end


function collect_snapshot(Population::Vector{MLP},
                          actions::Vector{Float64},
                          payoffs::Vector{Float64})

    hidden_sizes = [net.layers[2] for net in Population]

    # Mean response of each network to an opponent who last played C (=1) or D (=0)
    response_to_C = [clamp(forward(net, [1.0]), 0.0, 1.0) for net in Population]
    response_to_D = [clamp(forward(net, [0.0]), 0.0, 1.0) for net in Population]

    return (
        avg_coop = mean(actions),
        sd_coop = std(actions),
        avg_hidden_nodes = mean(hidden_sizes),
        avg_payoff = mean(payoffs),
        mean_response_to_C = mean(response_to_C),
        mean_response_to_D = mean(response_to_D),
        hidden_nodes_dist = copy(hidden_sizes)
    )
end

"""
Start with an initial Population, matching Pairs and all states = 1. Run the process for "nperiods" stage game. Every 
"store_interval" period, take a "snapshot".

Input:
- Population,
- Strategy_stored_Blank, 
- index_shifts, 
- Pairs, 
- otN, 
- beta, 
- gamma, 
- N,  
- epsilon, 
- payoff2b2, 
- nperiods, 
- store_interval, 
- mutation_rate, 
- match_types, 
- population_states, 
- max_n, 
- leaves, 
- w, 
- no_leave, 
- trimming

Output:
- match_type: a matrix counting the number of CC, CD and DD pairs
- population_states: an Array{Any} with a count of same strategies
- errors: how many times a mutation exceeds n_max
- leaves: leave count
"""


function pop_simulation(Population, Pairs, past_actions, otN, beta, gamma, N, epsilon, payoff2b2, nperiods, store_interval, mutation_rate, w)
    #initialize the quantities that are kept track of during the simulation 
    nsnaps = div(nperiods, store_interval)      
    avg_coop = zeros(Float64, nsnaps)
    sd_coop = zeros(Float64, nsnaps)
    avg_hidden_nodes = zeros(Float64, nsnaps)
    avg_payoff = zeros(Float64, nsnaps)
    mean_response_to_C = zeros(Float64, nsnaps)
    mean_response_to_D = zeros(Float64, nsnaps)
    hidden_nodes_dist = Vector{Vector{Int}}(undef, nsnaps)
    
    # Store the matching from the start of a block
    Pairs_Pre = Pairs; #the matching
    actions = zeros(Float64,N); #the actions that are played in the current round 
    
    # we take a snapshot of the population every store_interval stage-games. all steps in between two snapshots are in the "store_interval_runs()" function
    for t in 1:nsnaps
        Pairs_Pre, Population, Pairs, past_actions, actions, payoffs =
            store_interval_runs(store_interval, Population, Pairs, past_actions, otN,
                                beta, gamma, N, epsilon, payoff2b2, mutation_rate, w)

        snap = collect_snapshot(Population, actions, payoffs)

        avg_coop[t] = snap.avg_coop
        sd_coop[t] = snap.sd_coop
        avg_hidden_nodes[t] = snap.avg_hidden_nodes
        avg_payoff[t] = snap.avg_payoff
        mean_response_to_C[t] = snap.mean_response_to_C
        mean_response_to_D[t] = snap.mean_response_to_D
        hidden_nodes_dist[t] = snap.hidden_nodes_dist

        if t % 1000 == 0
            println(t / nsnaps)
        end
    end

    history = SimulationHistory(
        avg_coop,
        sd_coop,
        avg_hidden_nodes,
        avg_payoff,
        mean_response_to_C,
        mean_response_to_D,
        hidden_nodes_dist
    )
        
        # the leaves obect counts how many individuals play leave at each snapshot 
    return Population, Pairs, past_actions, actions, history #... all the metrics that are stored, over long time
end

## inner loop of the above function --> runs the one_period function store_interval times (one_period is one stage game) 
function store_interval_runs(store_interval, Population, Pairs, past_actions, otN, beta, gamma, N, epsilon,payoff2b2, mutation_rate,w)
    Pairs_Pre = Pairs
    actions = zeros(Float64, N)
    payoffs = zeros(Float64, N)

    for st in 1:store_interval
        Pairs_Pre = Pairs;
        Population, Pairs, past_actions, actions  = one_period(Population, Pairs, past_actions, otN, beta, gamma, N , epsilon,payoff2b2, mutation_rate,w);
    end

    return Pairs_Pre, Population, Pairs, past_actions, actions, payoffs
end  

## all things that happen in one stage game and before the next stage game
function one_period(Population::Vector{MLP}, 
                    Pairs::AbstractMatrix{<:Integer}, 
                    past_actions::Vector{Float64}, 
                    otN, 
                    beta::Float64, # Exogenous break probability 
                    gamma::Float64, # Selection event
                    N::Int,
                    epsilon::Float64,
                    payoff2b2, 
                    mutation_rate::Float64, 
                    w,) # Weight on payoff in the convex combination of payoff and 1

    #1) Determine what actions are played
    actions = nn_pop_actions(Population, past_actions, Pairs, epsilon)

    #2) what payoffs result
    # Assigns to each paired individual a stage-game payoff from payoff2b2, then mixes with baseline using w
    payoffs = nn_pop_payoffs(Pairs,actions,N,payoff2b2,w)
    
    #3) Update each individual's action
    nn_update_memory!(past_actions, actions)
    
    #4) who reproduces and who gets replaced  
    RI = rand(Binomial(N,(1-gamma)/2));#draw how many individuals are replaces --> divided by two, because the pop_selection function replaces both individuals in a pair at the same time, to minimize necessary break-ups per selection event
    
    if RI>0
        Pairs,Population,past_actions = pop_selection(payoffs, Pairs, Population, past_actions, N, 2*RI)
    end

    #5) exogenous break-ups
    Pairs = pop_breaks(Pairs, beta)

    #6) mutations #mutation_rate^2 is so close to zero that at max one mutation happens
    # TODO: check
    u = rand()
    if u < mutation_rate
        mut_ind = rand(1:N)
        rn_mut = rand()
        Population[mut_ind] = mutate_nn(Population[mut_ind], rn_mut)
    end

    #7) re-matching 
    # Finds singles, resets their states to 1, shuffles, pairs them up.
    Pairs, past_actions = pop_matching(Pairs,otN, past_actions)
   
    return Population, Pairs, past_actions, actions, payoffs
end

function simulation(payoff2b2, store_interval, nperiods, N, mutation_rate, epsilon, gamma, beta, w)
    # This is the wrapper for the pop_simulation function 
    # all the things that have to be declared once 
    @assert iseven(N)
    otN = collect(1:N);
    
    # here things that are stored 
    # TODO: add something meaningful here: what do we want to store?

    #initialize
    H = 5
    Population = [init_mlp([1, H, 1]) for _ in 1:N]
    past_actions = rand(N)
    Pairs = reshape(collect(1:N), (2,Int(N/2)));

    Population, Pairs, past_actions, actions = pop_simulation(Population, Pairs, past_actions, otN, beta, gamma,N, epsilon, payoff2b2, nperiods, store_interval, mutation_rate, w)
    return Population, Pairs, past_actions, actions
end 

end

function main()
    payoff2b2 = [3 0; 5 1]
    store_interval = 10
    nperiods = 1000
    N = 100
    mutation_rate = 0.01
    epsilon = 0.01
    gamma = 0.98
    beta = 0.95
    w = 0.5

    Population, Pairs, past_actions, actions, history = Simulations.simulation(
        payoff2b2,
        store_interval,
        nperiods,
        N,
        mutation_rate,
        epsilon,
        gamma,
        beta,
        w
    )

    println("Simulation finished.")
    println("Final pairs:")
    println(Pairs)
    println("Final average cooperation: ", history.avg_coop[end])
    println("Final average hidden nodes: ", history.avg_hidden_nodes[end])
    println("Final mean response to C: ", history.mean_response_to_C[end])
    println("Final mean response to D: ", history.mean_response_to_D[end])
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end