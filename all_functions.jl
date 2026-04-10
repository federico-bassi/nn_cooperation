# Intended actions in the current state
intended_actions(PS, state_v, index_shift) = PS[state_v + index_shift]

# Gather all the actions for the entire population, with the possibility of errors
pop_actions = function(N, epsilon, Pop, state_v, index_shift)
#...
end

# Gather all the payoffs for the population
pop_payoffs = function(Pairs, actions, N, payoff2b2, w)
#...
end

# Randomly break up some pairs
pop_breaks = function(Pairs, beta)
#...
end

# Given played actions, perform the state transition and gather for the whole population
state_transitions_pop5 = function (actions, Pairs, state_v_pre, Population)
#...
end

# Break up pairs in which the current action for at least one partner is "leave"
break_from_leave = function(Pairs, state_v, Population, index_shifts)
#...
end

chooseV_recycle = function(v,r)
#...
end

chooseV = function (v, r)
#...
end

# Perform a single mutation step on a strategy
mutation_basic_5d = function(mtd, rn, strategy, n)
#...
end

# Perform a selection step
Moran_selection = function(payoffs, N, RI, Pairs)
#...
end

# Perform the selection step and gather results and implications for pairs
pop_selection = function(payoffs, Pairs, Population, state_v, N, RI)
#...
end

# Perform rematching, and set states of rematched ones to one
pop_matching = function(Pairs, otN, state_v)
#...
end

# Do a full simulation
pop_simulation = function(Population, Stategy_stored_Blank, index_shifts, Pairs, otN, beta, gamma, N, epsilon, payoff2b2, nperiods, store_interval, mutation_rate, match_types, population_states, max_n, leaves, w, no_leave, trimming)
#...
end

# Inner loop of the above function
store_interval_runs = function(store_interval, Population, Pairs, state_v, otN, beta, gamma, N, max_n, epsilon, payoff2b2, index_shifts, mutation_rate, errors, w, no_leave, trimming)
#...
end

# All things that happen in one stage game and before the next stage game
one_period = function(Population, Pairs, state_v, otN, beta, gamma, N, max_n, epsilon, payoff2b2, index_shifts, mutation_rate, w, no_leave, trimming)
#...
end

# Exponential decay mutation