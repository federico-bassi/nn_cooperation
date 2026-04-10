"""
1. Objects

- Population: the finite state automaton is represented by a matrix where row indicates action in each state, rows 2-5 represent state transitions conditional on action profiles, columns represent states
- Pairs: a matrix 2*(N/2) representing current matchings
- state_v :  vector of automaton states for each individual

2. One stage game
    A. Action choice (pop_actions): intended action from automaton + trembles (ε)
    B. Payoffs: PD payoffs, convex combination with baseline payoffs (w)
    C. State transitions: update the automaton states given actions
    D. Selection (Moran process): with probability 1−γ, individuals are replaced
    E. Endogenous breakups: if someone playes "leave", the pair is dissolved
    F. Exogenous breakups: pairs dissolve with probability β²
    H. Mutation: with probability "mutation_rate" one individual mutates. Possible mutation:
        - Arrow rewiring
        - Action mutation
        - State deletion
        - State addition
        - Possibly formed by graph trimming or random reconnection.
    I. Rematching: singles are randomly rematched (pop_matching)

3. Mutation logic:
    A. mutation_basic_5d/_no_leave: mutation kernel
    B. exp_decay_mutation: number of mutations per event is geometrically distributed
    C. reduce_to_connected_graph/random_reconnect_graph: ensures automata are connected

4. Time aggregation:
    A. store_interval_runs: runs "one_period" repeated between two snapshots
    B. pop_simulation: the outer loop runs for "nperiods" and stores: (i) pair types, (ii) population state distribution, (iii) leave frequency, (iv) mutation overflow

5. Output and compression: strategies are compressed into integers (strat_5_to_number): to group identical automata and store population distributions efficiently (store_groupcounts5)
6. Wrapper: simulation(...) initializes
    - All-C population
    - Random matching
    - Storage containers
    - Calls pop_simulation
"""

#import Pkg;
#Pkg.add("Distributions")
#Pkg.add("Random")
#Pkg.add("StatsBase")
#Pkg.add("DataFrames")
#Pkg.add("FreqTables")
using Distributions
using Random
using StatsBase
using DataFrames
using FreqTables

"""
Returns the indented action in the current period for the whole population. Uses vectorized operations (faster)
"""
## Reads for every individual, what are the intended actions in the current state, i.e. the actions if no error happens 
## Error happens on C-D, not on leave  
intended_actions(PS,state_v,index_shifts) = PS[state_v+index_shifts]; ## PS = Population[:,1,:]' #THE TRANSPOSE IS IMPORTANT


"""
Computes the actual actions, including implementation errors. 
"""
pop_actions = function(N,epsilon,Pop,state_v,index_shifts)
    i_actions = intended_actions(Pop[:,1,:]', state_v,index_shifts); #which is a vector of 0,1,2, where 0 = leave, 1 = cooperate, 2 = defect
    # Decide whether there is error implementation for individual i
    play_errors = rand(N).<epsilon; 
    # errors can only act on the fields that are 1 or 2. If play errors is true, action 1 becomes 2 and action 2 becomes 1
    actions = i_actions  + ((i_actions.==1)-(i_actions.==2)).*play_errors;
    return actions
end


"""
Computes the payoffs in the population

Inputs:
- Pairs: 2*(N/2) matrix, each column is a matched (i,j) pair. 
    E.g. the matrix [1 2 3
                     4 5 6]
        says that the pairs are (1,4), (2,5) and (3,6)
- actions: length-N vector with entries {0,1,2}
- N: population size
- payoff2b2: 2*2 payoff matrix
- w in [0,1]: weight on the payoff of the interaction, controls selection strength

Outputs:
- A 1*N matrix with payoff for each individual in the population
"""
pop_payoffs  = function(Pairs,actions,N,payoff2b2,w)
    # Initialise a vector of zeros
    payoffs = zeros(1,N);
    # Loop over all matched pairs (if they are matched, they are not going to play "leave").
    # size(Pairs, 2) returns the size of Pairs along dimension 2 (columns), i.e. the number of pairs in the population (N/2)
    for i in 1:size(Pairs,2)
        # Index i will represent the column (index of the pair). For a given pair, retrieve the index of individual a with Pairs[1,i]
        # and that of individual b with Pairs[2,i]. You can then go to the "actions" vector and retrieve the action taken by a and b.
        # Finally go to to payoff2b2 and retrieve the entry [actions[a], actions[b]] as payoff for the first player, and [actions[b], actions[a]]
        # as payoff for the second player.
        payoffs[Pairs[1,i]] = payoff2b2[actions[Pairs[1,i]], actions[Pairs[2,i]]];
        payoffs[Pairs[2,i]] = payoff2b2[actions[Pairs[2,i]], actions[Pairs[1,i]]];
    end
    # Payoffs are convex combinations of 1 and the payoff from the matrix
    payoffs = (1-w).*ones(1,N) + (w).*payoffs;
    return payoffs
end

"""
Randomly breaks up some pairs.
Input:
- Pairs: matrix 2*(N/2) containing the pairs
- beta: Beta is taken as the per-individual continuation probability, so beta**2 is the probability that both individuals in a pair survive

Output:
- Pairs without columns
"""
pop_breaks = function(Pairs,beta)
    # Take the number of pairs and draw one uniform random number per pair
    r = rand(size(Pairs,2));
    # In the matrix Pairs, take the columns that have r lower than Beta**2
    Pairs = Pairs[:, (r.<(beta^2))]; # the events are i.i.d.
    return Pairs
end

"""
Perform the state transition for the whole population.

Input:
- actions
- Pairs
- state_v_pre
- Population

Output:
- state_v
"""
state_transitions_pop5 = function(actions,Pairs, state_v_pre, Population) #use this only on those pairs that didn't break, so actions(pairs) only contain 1's and 2's
    state_v = copy(state_v_pre)
    # Loop over the pairs and compute the new state for each individual in the pair
    for i in 1:size(Pairs,2)
        # The state for the first individual in the pair will be determined by: (i) her state, (ii) her action and the other person's action
        # Given that individual i performs action a_i and individual j performs action a_j, the formula: 2*(a_i-1)+a_j+1 returns the row that
        # tells you which is the new state to which you need to go depending on the realised action (CC, CD, DC, DD)
        state_v[Pairs[1,i]] = Population[Pairs[1,i],2*(actions[Pairs[1,i]]-1)+actions[Pairs[2,i]]+1,state_v_pre[Pairs[1,i]]];
        state_v[Pairs[2,i]] = Population[Pairs[2,i],2*(actions[Pairs[2,i]]-1)+actions[Pairs[1,i]]+1,state_v_pre[Pairs[2,i]]];
    end
    return state_v
end

"""
After states have been updated, player's automaton might prescribe leave. This function checks intended action, checks pairs and remove
pairs where at least one member intends to leave.
"""
## break up pairs in which the current action for at least one partner is 'leave'
break_from_leave  = function(Pairs,state_v,Population,index_shifts)
    ## part that deletes the pairs that have a leaving individual in them
    IA = intended_actions(Population[:,1,:]',state_v,index_shifts);
    # Create a new array with the same shape as Pairs but with the entries of IA
    t_pre = IA[Pairs];
    # Sum the rows of t_pre where t_pre == 0, convert it to vector
    t_pre = vec(sum((t_pre.==0),dims=1));
    # Keep only the pairs where no one intends to leave
    Pairs = Pairs[:,findall(t_pre.==0)];
return Pairs
end

"""
Draw discrete events from a cdf and then recycle the same uniform random number ot generate a second "approximately" uniform on [0,1] inside
the selected.

Inputs:
- v: a vector representing a cdf grid
- r: a draw from Uniform(0,1)

Outputs:
- index: which event/interval was selected
- r_new: a recycled uniform number within the chosen interval
"""
## use random number in normalized, stacked vector to map draw from a uniform [0,1] to an event, and return a random number
## where in a given interval a randomly drawn number lies is uncorrelated with whether or not it lies in this interval. So this exact location can be used as a new random number.
## (the rationale for doing this, is that drawing random numbers takes time, and that we have few events 
## -- all with probabilities that are large compared to the precision of the random number, so re-using the random number is computationally cheaper, and inconsequential for the result)
chooseV_recycle = function(v,r) #v must already represent a cdf, with first element 0, and last 1.
    # Loop over the numbers from 1 to length of v (indices), then find the first index x such as v[x] is bigger than r
    index =  findfirst( x -> v[x] > r, 1:length(v));
    # Compute the new random number
    # TODO : check, shouldn't the denominator be (v[index]-v[index-1])
    r_new = (r-v[index-1])/v[index];
    index = index-1;
    return index, r_new
end

## same without returning a random number
chooseV = function(v,r) #v must already represent a cdf
   index =  findfirst( x -> v[x] > r, 1:length(v));
    return index
end


"
Performs a mutation step on a strategy. Four types of different mutations:
1. Rewire an Arrow
2. Change an action
3. Delete a state
4. Add a state

Inputs:
- strategy: 5*max_n matrix
- n: current number of active states
- rn: uniform draw
- mtd: thresholds for mutation-type probabilities with interpretation: mtd[1] - arrow mutation, mtd[2] - arrow or state-output mutation, mtd[3] - arrow/output/deletion, else - addition

Outputs:
- mutated strategis
"
mutation_basic_5d = function(mtd,rn,strategy,n) # 5d indicates that the strategy is 5 by max_n -- there is an equivalent version for 3 by max_n matrices
    #arrow mutation (as described in thesis)
    if rn < mtd[1]
        # Only enter when there is more then 1 state
        if n!=1
            rn = rn/mtd[1];
            # Choose u.a.r. which is the index of mutation (idm) (this is a linear index --> below, transform this linear index into cartesian)
            # Use chooseV_recycle where the range goes from 0 to 1 with step equal to 1/4*n (where 4*n is the number of arrows)
            # idm becomes a number like 1,2,...,4n indicating which arrow was chosen
            idm,rn = chooseV_recycle(collect(0:(1/(4*n)):1),rn);
            # Convert idm into row and column
            idmrow = mod(idm,4)+2; # first part takes values 0,1,2,3 --> map to 2,3,4,5
            idmcol = convert(Int64,floor((idm-1)/4)+1);  # like this, it takes values 1,...,n //  example n=2 --> idm = 1,2,3,4 map to 1 --> 5,6,7,8 map to 2 
            push = convert(Int64,ceil((n-1)*rn)); # we are in Z modulo n. we move between 1 and n-1 steps --> that puts us to a proper different spot in Z modulo n 
            # strategy[idmrow,idmcol]+push-1 --> subtract one to move from 1-N to Z modulo n, and then add one to map back 
            strategy[idmrow,idmcol] = 1+mod(strategy[idmrow,idmcol]+push-1,n);# could be +1, +2,... modn (can only point to existing states)
        end
    #state-output mutation (as described in thesis)
    elseif rn<mtd[2] 
        rn = (rn-mtd[1])/(mtd[2]-mtd[1]);
        idm,rn=chooseV_recycle(collect(0:(1/(n)):1),rn) #which state's action is chosen 
        if idm !=1 #only really go through with this, if the mutation doesn't change the first state to a zero
        strategy[1,idm] = mod(strategy[1,idm]+ceil(2*rn),3) #could be 0,1,2 --except the one that already is specified there // all permitted
        else
        strategy[1,idm] = mod(strategy[1,idm],2)+1; #because 0 (leave) is not permitted in the first state. First, map back to {0,1} then add one, modulo it by 2, then map back to {1,2}, so it becomes  mod(strategy[1,idm],2)+1-1, which cancels
        end
    #state-deletion  (as described in thesis)
    elseif rn< mtd[3] 
        if n>1
        rn = (rn-mtd[2])/(mtd[3]-mtd[2]);
        #let's make it simple and make sure that state 1 is not deleted
        idm,rn=chooseV_recycle(collect(0:(1/(n-1)):1),rn);#which state is chosen to be deleted
        idm = idm+1 #because we excluded the first state
        # redirect all pointers randomly 
        point_to_v = union(1:(idm-1),(idm+1):n); #possible states to point to 
        for i in union(1:(idm-1),(idm+1):n)
            for j in 2:5
                if strategy[j,i]==idm
                        strategy[j,i] = rand(point_to_v); 
                end
            end
        end         
        #now the idm'th column is deleted, and all numbers in strategy[2:end,:]>idm are reduced by one
            if idm==size(strategy,2)
                strategy[:,idm].=0;
            else
                strategy[:,idm:end-1] = strategy[:,idm+1:end];                
            end
            strategy[2:end,:]  =strategy[2:end,:].- (strategy[2:end,:].>idm);
        end
    else #state-addition  (as described in thesis)
        rn = (rn-mtd[3])/(1- mtd[3])
        #add state: this is the most involved one
        n = n+1;
        #need to draw 4 numbers 
        index1,rn = chooseV_recycle(collect(0:1/n:1),rn);
        index2,rn = chooseV_recycle(collect(0:1/n:1),rn);
        index3,rn = chooseV_recycle(collect(0:1/n:1),rn);
        index4,rn = chooseV_recycle(collect(0:1/n:1),rn);
        strategy[2,n]=index1;
        strategy[3,n]=index2;
        strategy[4,n]=index3;
        strategy[5,n]=index4; 
        strategy[1,n] = mod(ceil(3*rn),3); #lastly we draw an action u.a.r. 
        # now we have wiring from the state. but we also need wiring to the state
        # we randomly draw one element that links there
        # we're still exploiting the same random number
        rn = mod(rn,1/3);
        idm = chooseV(0:1/(4*(n-1)):1,rn);
        idmrow = mod(idm,4)+2; # first part takes values 0,1,2,3 --> map to 2,3,4,5
        idmcol = convert(Int64,floor((idm-1)/4)+1);  # like this, it takes values 1,...,n //  example n=2 --> idm = 1,2,3,4 map to 1 --> 5,6,7,8 map to 2
        strategy[idmrow,idmcol] = n;    
    end
    return strategy
end



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

Moran_selection =  function(payoffs,N, RI,Pairs) #RI the number of replacing individuals
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
Actually performs the selection: updates the population, reset states, fixes the pairing structure
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
pop_selection = function(payoffs,Pairs,Population,state_v, N, RI)
    ## new_inds are the indivuals that are being replaced
    ## repstr are the reproducing strategies --> so in the population object the new_inds will have their strategy updated to the strategies in the repstr object
    new_inds, repstr  = Moran_selection(payoffs,N, RI,Pairs); #the new individuals
    
    # Reset the states of new individuals
    state_v[new_inds] .= 1;
    Population[new_inds,:,:] = Population[repstr,:,:]; 

    # Create two booleans a and b with length = number of pairs where a tells if first member of Pairs is among "new indices"
    # i.e. indices that are to be replaced
    a = in(new_inds).(Pairs[1,:]);
    b = in(new_inds).(Pairs[2,:]);
    
    # Sum the two boolean vectors (0 only if both members of the pair are not to be replaced)
    changed = (a.+b);

    # Keep only pairs for which changed is 0
    Pairs = Pairs[:,(changed.==0)]
    
    return Pairs, Population, state_v 
end


## perform rematching, and set states of rematched ones to one
pop_matching = function(Pairs,otN, state_v)  
    # Compare otN (1:N) with individuals in Pairs, then return the difference
    singles = setdiff(otN,vec(Pairs));
    
    # States the initial state of the singles to 1
    state_v[singles] .= 1;

    # Randomly permute the singles
    singles = shuffle(singles);

    # Horizontally concatenate Pairs with a reshaped version of singles (2 rows, as many columns as needed)
    Pairs = hcat(Pairs,reshape(singles, 2,:));
    return Pairs, state_v
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


pop_simulation = function(Population,Strategy_stored_Blank, index_shifts, Pairs, otN, beta, gamma, N,  epsilon, payoff2b2, nperiods, store_interval, mutation_rate, match_types, population_states, max_n, leaves, w, no_leave, trimming)
    # define some very frequently used objects (most are the same as in the data_types_back_and_forths.jl file)
    base_s5tn_v1 = 3;
    # Build powers of 3
    s5tn_v1 = base_s5tn_v1.^collect(0:(max_n-1)); 
    # Build powers of n+1
    s5tn_v2 = (max_n+1).^collect(0:(max_n-1)); # because it could be empty (zero)

    #initialize the quantities that are kept track of during the simulation 
    
    # Make a vector of N ones (every automaton starts in state 1)
    state_v = ones(Int64,N); #what states the automata are in 
    # Store the matching from the start of a block
    Pairs_Pre = Pairs; #the matching
    actions = zeros(Int64,N); #the actions that are played in the current round 
    errors = zeros(convert(Int64,(nperiods/store_interval))); # this is simply a tool to keep track of whether max_n is chosen appropriately --> if strategies mutate to something larger than max_n the mutation is aborted and this error variable is increased by one

    # we take a snapshot of the population every store_interval stage-games. all steps in between two snapshots are in the "store_interval_runs()" function
        for t in 1:convert(Int64,(nperiods/store_interval))
            Pairs_Pre, Population, Pairs, state_v, actions, errors_i, leave_count = store_interval_runs(store_interval, Population, Pairs, state_v, otN, beta, gamma, N, max_n , epsilon,payoff2b2, index_shifts, mutation_rate,errors,w,no_leave,trimming)
            # Record errors
            errors[t] = errors_i;
            # Record in column t of "match_types": (1) number of CC pairs, (2) number of CD pairs, (3) number of DD pairs 
            match_types[:,t] = pair_types(Pairs_Pre,actions); #because actions are recoreded at the beginning of period, and not later 
            # Record what strategies exist and how common they are
            population_states[t] = store_groupcounts5(Population,max_n,N,Strategy_stored_Blank,base_s5tn_v1,s5tn_v1,s5tn_v2);
            # Record the number of leave
            leaves[t] = leave_count;
            # Print progress every 1000 snapshots
            if t%1000 ==0
                println(t/(nperiods/store_interval))
            end
        end
        # Collapse erros into a single number
        errors = sum(errors)

        # the leaves obect counts how many individuals play leave at each snapshot 
    return match_types, population_states, errors, leaves#... all the metrics that are stored, over long time
end


## inner loop of the above function --> runs the one_period function store_interval times (one_period is one stage game) 
store_interval_runs = function(store_interval, Population, Pairs, state_v, otN, beta, gamma, N, max_n , epsilon,payoff2b2, index_shifts, mutation_rate,errors,w,no_leave,trimming)
    Pairs_Pre = Pairs
    actions = zeros(Int64,N)
    leave_count = 0
    errors = zeros(convert(Int64,store_interval));
    for st in 1:store_interval
        Pairs_Pre = Pairs;
        Population, Pairs, state_v, actions_i, errors_i, leaves_i =one_period(Population, Pairs, state_v, otN, beta, gamma, N, max_n , epsilon,payoff2b2, index_shifts, mutation_rate,w,no_leave,trimming);
        errors[st] = errors_i;
        actions = actions_i;
        leave_count = leaves_i;
    end
    errors = sum(errors)
    return Pairs_Pre, Population, Pairs, state_v, actions, errors, leave_count
end    
    


## all things that happen in one stage game and before the next stage game
one_period = function(Population, Pairs, state_v, otN, beta, gamma, N, max_n , epsilon,payoff2b2, index_shifts, mutation_rate,w,no_leave,trimming)
    count_errors = 0;
    
    #1) what actions are played
    # Returns a length-N vectors in {0,1,2} and includes errors with probability epsilon, but only flips C with D
    actions = pop_actions(N,epsilon,Population,state_v,index_shifts);
    
    #2) what payoffs result
    # Assigns to each paired individual a stage-game payoff from payoff2b2, then mixes with baseline using w
    payoffs = pop_payoffs(Pairs,actions,N,payoff2b2,w);
    # Update each individual's current state based on the outcome (CC/CD/DC/DD) in the pair
    state_v  = state_transitions_pop5(actions,Pairs,state_v,Population);
    
    #3) who reproduces and who gets replaced  
    RI = rand(Binomial(N,(1-gamma)/2));#draw how many individuals are replaces --> divided by two, because the pop_selection function replaces both individuals in a pair at the same time, to minimize necessary break-ups per selection event
    if RI>0
        Pairs,Population,state_v= pop_selection(payoffs,Pairs,Population,state_v, N, 2*RI);
    end
    
    #4) leave-events -- if no_leave==1, nothing happens, because no state-output is zero
    count_pre = size(Pairs,2);
    Pairs = break_from_leave(Pairs,state_v,Population,index_shifts);
    leaves_i = count_pre - size(Pairs,2);

    #5) exogenous break-ups
    Pairs = pop_breaks(Pairs,beta); #the state_v of the broken ones is not updated yet, but will be in the matching function

    #6) mutations #mutation_rate^2 is so close to zero that at max one mutation happens
    rn  = rand(1)[1];#draw a random number
    if rn<mutation_rate
        rn = rn/mutation_rate; 
        mut_ind = convert(Int64, ceil(rn*N)); #randomly choose an individual that mutates
        
        # n = length of this strategy
        # Go to the second row of Population[mut_ind,:,:], check whether the last column is 0. If it is not, then n=max_n,
        # if it is, find the index of the last non-zero column
        if Population[mut_ind,2,end]!=0 
            n = max_n;
        else
            n =-1+ findfirst( x -> Population[mut_ind,2,x] ==0, 1:max_n);
        end

        # weight_increase = likelihood(adding state)/(likelihood(adding state)+likelihood(deleting state))
        # biased towards state deletion
        # this is set to zero if n==max_n
        weight_increase = 0.45;
        if n == max_n
            count_errors = 1;
            weight_increase= 0;
        end
        
        decay = 0.7; #number of mutations follows an exponential distribution with this decay parameter
        new_strategy = exp_decay_mutation!(rn,decay,Population[mut_ind,:,:],weight_increase,max_n,no_leave);

        # the different versions of the mutation kernel 
        # trimming == 1: delete unreacheable states
        if trimming == 1
            Population[mut_ind,:,:]  = reduce_to_connected_graph(new_strategy);
        # trimming == 0: reconnect randomly until reacheable
        elseif trimming ==0
            Population[mut_ind,:,:]  = random_reconnect_graph(new_strategy);
        # keep as it is
        else
            Population[mut_ind,:,:]  = new_strategy;
        end
        # mut_ind is eliminated from Pairs, if it is in a pair
        Pairs = Pairs[:,Pairs[1,:].!= mut_ind];
        Pairs = Pairs[:,Pairs[2,:].!= mut_ind];
    end

    #7) re-matching 
    # Finds singles, resets their states to 1, shuffles, pairs them up.
    Pairs,state_v = pop_matching(Pairs,otN,state_v);

    return Population, Pairs, state_v, actions, count_errors, leaves_i
end



## exponential decay mutation 
exp_decay_mutation! = function(rn, decay, strategy,weight_increase,max_n,no_leave)
     for i = 1:draw_exponential_decay(decay)
        rn = rand(1)[1]
        if strategy[2,end]!=0 
            n = max_n;
        else
            n =-1+ findfirst( x -> strategy[2,x] ==0, 1:max_n);
        end
        # THESE ARE THE WEIGHTS OF THE DIFFERENT TYPES OF MUTATIONS

        mtd = [0.2 0.4 0.4+(0.6*(1-weight_increase)) 1];
        
        
        if n == max_n
            mtd  = [0.2 0.4 1]
        end

        if no_leave == 0
            strategy = mutation_basic_5d(mtd,rn,strategy,n);
        else
            strategy = mutation_basic_5d_no_leave(mtd,rn,strategy,n);
        end
    end
    return strategy
end


draw_exponential_decay = function(decay)
    r = rand(1)[1]
    i = 1;
    while r>1-((decay)^i)
        i = i+1;
    end
    return i
end

store_groupcounts5 = function(population,max_n,N,Strategy_stored_Blank,base_s5tn_v1,s5tn_v1,s5tn_v2)
    # groups all individuals with identical strategy to reduce required storage space 
    Strategy_stored = deepcopy(Strategy_stored_Blank);
    for i in 1:N
        Strategy_stored[i,:]=strat_5_to_number(population[i,:,:],max_n,s5tn_v1,s5tn_v2);
    end
    return combine(groupby(Strategy_stored, [:x1, :x2, :x3, :x4, :x5]), nrow)
end


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

strat_5_to_number = function(strat_,max_n,s5tn_v1,s5tn_v2)
# as in data_types_back_and_forths
    strat = deepcopy(strat_) 
    s = zeros(UInt64, 5);
    s[1]=strat[1,:]'*s5tn_v1
    s[2]=strat[2,:]'*s5tn_v2
    s[3]=strat[3,:]'*s5tn_v2
    s[4]=strat[4,:]'*s5tn_v2
    s[5]=strat[5,:]'*s5tn_v2
    return s
end


simulation = function(payoff2b2,max_n,store_interval,nperiods,N,mutation_rate,epsilon,gamma,beta,w,no_leave,trimming)
    # This is the wrapper for the pop_simulation function 
    # all the things that have to be declared once 
    Strategy_stored_Blank = DataFrame(zeros(Int64,N,5),:auto);
    otN = collect(1:N);
    index_shifts = [ (x-1)*max_n for x in 1:N ]; 
    
    # here things that are stored 
    ntime = convert(Int128,nperiods/store_interval)
    match_types = zeros(3,ntime);
    population_states = Array{Any}(undef,ntime);
    leaves = zeros(ntime);

    #initialize
    Population = zeros(Int64,N,5,max_n); #start with allC
    Population[:,:,1].= 1; 
    Pairs = reshape(collect(1:N), (2,Int(N/2)));
    match_types, population_states, errors , leaves= pop_simulation(Population,Strategy_stored_Blank, index_shifts, Pairs, otN, beta, gamma, N, epsilon,payoff2b2, nperiods, store_interval, mutation_rate, match_types, population_states,max_n, leaves,w,no_leave,trimming);
        return match_types, population_states, errors, leaves
end






reduce_to_connected_graph = function(strategy)     # delets states in an automaton that cannot be reached 
    reachable ,n = identify_disconnections_graph(strategy);
    # now, carefully delete all columns that are disconnected, and change indices accordingly (nothing has pointed to these columns previously)
    if (length(reachable))<n
        #indices that need to be deleted 
        d_indices = setdiff(collect(1:n), reachable)
        #now the idm'th column is deleted, and all numbers in strategy[2:end,:]>idm are reduced by one
        for idm = sort(d_indices, rev=true)
            if idm==size(strategy,2)
                strategy[:,idm].=0;
            else
                strategy[:,idm:end-1] = strategy[:,idm+1:end];
                strategy[:,size(strategy,2)].=0;
            end
            strategy[2:end,:]  =strategy[2:end,:].- (strategy[2:end,:].>idm);
        end
    end
    return (strategy)
end



identify_disconnections_graph  = function(strategy) #identifies components of the graph
    #works on 3 and 5 strategies
        if strategy[2,end]!=0 
            n = size(strategy,2);
        else
            n =-1+ findfirst( x -> strategy[2,x] ==0, 1:size(strategy,2)); # n states are defined 
        end
        reachable = unique(union(1,unique(strategy[2:end,1]))) #those are reachable from state 1
        i=1; #the index in the reachable_stay vector 
        while (length(reachable))<n && (i<length(reachable))  #if we haven't identified all states and if and we still have reachable states to consider
            i = i+1;
            a = strategy[2:end,reachable[i]]; 
            ap = setdiff(a, reachable);
            reachable = append!(reachable,ap);
        end
        return reachable ,n
end



random_reconnect_graph  = function(strategy)
    reachable ,n = identify_disconnections_graph(strategy);
    strategy  = random_reconnect_graph_loop(strategy, reachable, n);
    return (strategy)
end

random_reconnect_graph_loop = function(strategy, reachable, n)
    while length(reachable)<n
        strategy,reachable = random_reconnection_step(strategy, reachable, n)
    end
    return strategy
end

random_reconnection_step = function(strategy, reachable, n)
    # we have an unconnected graph (otherwise the function wouldn't be called)
    d_indices = setdiff(collect(1:n), reachable)
    d_indices = shuffle(d_indices)
    k = length(reachable)
    strategy_new = deepcopy(strategy)
    arrow = shuffle(collect(1:((size(strategy,1)-1)*k)))[1] #choose a random arrow to point there
    @views strategy_new[2:end, reachable][arrow] = d_indices[1]; #the pointing has happened
    reachable_new ,n = identify_disconnections_graph(strategy_new)
    if length(setdiff(reachable,reachable_new))==0 #if there is nothing in reachable that isn't in reachable_new
        return strategy_new, reachable_new
    else
        return strategy, reachable
    end
end



mutation_basic_5d_no_leave = function(mtd,rn,strategy,n) #same as the one with leaving, just that no state output can be changed into a 0
    if rn < mtd[1] #arrow mutation
        if n!=1
            rn = rn/mtd[1];
            idm,rn=chooseV_recycle(collect(0:(1/(4*n)):1),rn); 
            idmrow = mod(idm,4)+2; 
            idmcol = convert(Int64,floor((idm-1)/4)+1);
            push = convert(Int64,ceil((n-1)*rn)); 
            strategy[idmrow,idmcol] = 1+mod(strategy[idmrow,idmcol]+push-1,n);
        end
    elseif rn<mtd[2]
        rn = (rn-mtd[1])/(mtd[2]-mtd[1]);
        idm,rn=chooseV_recycle(collect(0:(1/(n)):1),rn)  
        #####################version without leaves ##################
        strategy[1,idm] = mod(strategy[1,idm],2)+1;
    elseif rn< mtd[3]
        if n>1
        rn = (rn-mtd[2])/(mtd[3]-mtd[2]);
        idm,rn=chooseV_recycle(collect(0:(1/(n-1)):1),rn);
        idm = idm+1 

        point_to_v = union(1:(idm-1),(idm+1):n);  
        for i in union(1:(idm-1),(idm+1):n)
            for j in 2:5
                if strategy[j,i]==idm
                        strategy[j,i] = rand(point_to_v); 
                end
            end
        end    
            if idm==size(strategy,2)
                strategy[:,idm].=0;
            else
                strategy[:,idm:end-1] = strategy[:,idm+1:end];                
            end
            strategy[2:end,:]  =strategy[2:end,:].- (strategy[2:end,:].>idm);
        end
    else
        rn = (rn-mtd[3])/(1- mtd[3])
        n = n+1;
        index1,rn = chooseV_recycle(collect(0:1/n:1),rn);
        index2,rn = chooseV_recycle(collect(0:1/n:1),rn);
        index3,rn = chooseV_recycle(collect(0:1/n:1),rn);
        index4,rn = chooseV_recycle(collect(0:1/n:1),rn);
        strategy[2,n]=index1;
        strategy[3,n]=index2;
        strategy[4,n]=index3;
        strategy[5,n]=index4; 
        ################ adapted in no-leave-simulation###############
        strategy[1,n] =1+ mod(ceil(2*rn),2); 
        rn = mod(rn,1/3);
        idm = chooseV(0:1/(4*(n-1)):1,rn);
        idmrow = mod(idm,4)+2;
        idmcol = convert(Int64,floor((idm-1)/4)+1);
        strategy[idmrow,idmcol] = n;    
    end
    return strategy
end

strat_3_to_number = function(strat_,max_n,s5tn_v1,s5tn_v2)
# same as in data_types_back_and_forths.jl file
    strat = copy(strat_)
    s = zeros(UInt64, 3);
    s[1]=strat[1,:]'*s5tn_v1
    s[2]=strat[2,:]'*s5tn_v2
    s[3]=strat[3,:]'*s5tn_v2
    return s
end