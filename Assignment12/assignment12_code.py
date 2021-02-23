from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List,Optional,Iterable,TypeVar,Mapping,Dict
import numpy as np
from rl.distribution import Constant, Categorical
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian,Uniform,Constant
from rl.markov_decision_process import MarkovDecisionProcess, Policy
import rl.markov_process as mp
from rl.returns import returns
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite,InventoryState

from rl.monte_carlo import mc_prediction
from rl.td import td_prediction
from rl.function_approx import Tabular
from rl.function_approx import FunctionApprox
from rl.markov_process import FiniteMarkovRewardProcess
import matplotlib.pyplot as plt

S = TypeVar('S')
A = TypeVar('A')

#Problem 1
def n_step_bootstrap_tabular(
        transitions: Iterable[mp.TransitionStep[S]],
        states:List[S],
        gamma: float,
        num_transitions: float = 10000,
        initial_lr:float = 0.1,
        half_life: float = 1000.0,
        exponent: float = 0.5,
        n:int = 1
) -> Mapping[S,float]:

    v:Mapping[S,float] = {}
    counts_per_state:Mapping[S,int] = {} #We need this for the learning rate
    #Initialization step
    for state in states:
        v[state] = 0.
        counts_per_state[state] = 0
    count_transitions = 0
    list_states:List[S] = []
    list_rewards:List[float] = []
    for transition in transitions:
        #We choose the gather in an ordered list all the states and transitions
        #To make processing easier after that
        list_states.append(transition.state)
        list_rewards.append(transition.reward)
        count_transitions+=1
        if count_transitions>num_transitions:
            break
    for i in range(len(list_states)):
        state:S = list_states[i]
        #Below we define the learning rate
        counts_per_state[state] += 1
        #lr = 1/counts_per_state[state]
        #lr = 0.1
        lr = initial_lr / (1 + ((counts_per_state[state]-1)/half_life)**exponent)
        #Below we compute Gt_n
        last_index = min(i+n,len(list_states)-1)
        Gt_n:float = 0.
        for j in range(i,last_index):
            Gt_n += list_rewards[j]*gamma**(j-i)
        Gt_n += gamma**(last_index-i)*v[list_states[last_index]]
        v[state] = v[state] * lr*(Gt_n-v[state])
    return v

def n_step_bootstrap_fapprox(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        gamma: float,
        num_transitions: float = 10000,
        initial_lr:float = 0.1,
        half_life: float = 1000.0,
        exponent: float = 0.5,
        n:int = 1
) -> FunctionApprox[S]:
    
    v:FunctionApprox[S] = approx_0
    counts_per_state:Mapping[S,int] = {} #We need this for the learning rate
    list_states:List[S] = []
    list_rewards:List[float] = []
    count_transitions = 0
    for transition in transitions:
        #We choose the gather in a list all the states and transitions
        #To make processing easier after that
        list_states.append(transition.state)
        list_rewards.append(transition.reward)
        count_transitions+=1
        if count_transitions>num_transitions:
            break
    
    for i in range(len(list_states)):
        state:S = list_states[i]
        #Below we define the learning rate
        if state in counts_per_state:
            counts_per_state[state] += 1
        else:
            counts_per_state[state] = 1
        lr = initial_lr / (1 + ((counts_per_state[state]-1)/half_life)**exponent)
        #Below we compute Gt_n
        last_index = min(i+n,len(list_states)-1)
        Gt_n:float = 0.
        for j in range(i,last_index):
            Gt_n += list_rewards[j]*gamma**(j-i)
        Gt_n += gamma**(last_index-i)*v(list_states[last_index])
        #Then we can update our function approximation
        v = v.update([state,v(state) * lr*(Gt_n-v(state))])
    return v

#Problem 2
def td_lambda_tabular(
        transitions: Iterable[mp.TransitionStep[S]],
        states:List[S],
        gamma: float,
        lamb:float,
        num_transitions: float = 10000,
        lr:float = 0.1,
) -> Mapping[S,float]:
    memory_func:Mapping[S,float] = {}
    v:Mapping[S,float] = {}
    #We initialize values
    for state in states:
        v[state] = 0.
        memory_func[state] = 0.
    count_transitions = 0
    for transition in transitions:
        count_transitions+=1
        if count_transitions>num_transitions:
            break
        state:S = transition.state
        reward:float = transition.reward
        next_state:S = transition.next_state
        for k in memory_func.keys():
            memory_func[k] = gamma*lamb*memory_func[k] +(state == k)
        update:float = lr*(reward+gamma*v[next_state]-v[state])
        for s in v:
            v[s] = v[s]+update*memory_func[s]
    return v

def td_lambda_approx(
        transitions: Iterable[mp.TransitionStep[S]],
        approx_0: FunctionApprox[S],
        gamma: float,
        lamb:float,
        num_transitions: float = 10000,
        lr:float = 0.01,
) -> FunctionApprox[S]:
    memory_func:Mapping[S,float] = {}
    v:FunctionApprox[S] = approx_0
    #We initialize values
    count_transitions = 0
    for transition in transitions:
        count_transitions+=1
        if count_transitions>num_transitions:
            break
        state:S = transition.state
        reward:float = transition.reward
        next_state:S = transition.next_state
        for k in memory_func.keys():
            memory_func[k] = gamma*lamb*memory_func[k] +(state == k)
        if state not in memory_func.keys():
            memory_func[state] = 1
        
        update:float = lr*(reward+gamma*v(next_state)-v(state))
        list_updates:List[Tuple[S,float]] = []
        for s in memory_func.keys():
            list_updates.append((s,v(s)+update*memory_func[s]))
        v = v.update(list_updates)
    return v

#Problem 4 -- We need some utility functions to plot graphs of convergence for different values of lambda
def get_rmse(true_value_func:Mapping[S,float],
             predicted_value_func:Mapping[S,float])->float:
    value:float = 0
    count:int = len(true_value_func)
    for state in true_value_func.keys():
        value += (true_value_func[state]-predicted_value_func[state])**2
    return np.sqrt(value/count)
        
if __name__ == '__main__':

    print("Testing our implementations for Problems 1 and 2 and solving problem 4")
    print("We're using the inventory problem like in Assignment 11")
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0
    user_gamma = 0.9
    lamb:float = 0.5
    num_transitions:int = 10000
    lr:float = 0.01

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )
    print("Value Function by Policy Evaluation")
    print("--------------")
    print(si_mrp.display_value_function(gamma=user_gamma))
    #true_value_func = si_mrp.get_value_function_vec(gamma = user_gamma)
    true_value_func = {InventoryState(on_hand=0, on_order=0): -33.74685689376906,
                         InventoryState(on_hand=0, on_order=1): -26.690796783542474,
                         InventoryState(on_hand=0, on_order=2): -26.69295420262949,
                         InventoryState(on_hand=1, on_order=0): -27.60677614368549,
                         InventoryState(on_hand=1, on_order=1): -27.88738256958404,
                         InventoryState(on_hand=2, on_order=0): -28.44761732927098}
    print()
    
    states:List[InventoryState] = si_mrp.non_terminal_states
    start_state_distrib: Categorical[InventoryState] = Categorical({i:1 for i in states})
    simulation_episodes = si_mrp.reward_traces(start_state_distrib)
    simulation_transitions = si_mrp.simulate_reward(start_state_distrib)
    approx_0 = Tabular({i : 0 for i in states})
    
    print("Value Function by TD(lambda)")
    value_function = td_lambda_tabular(simulation_transitions,
                                      states,
                                      user_gamma,
                                      lamb,
                                      num_transitions,
                                      lr)
    print(value_function)
    
    list_rmses:List[float] = []
    lambdas = np.linspace(0,1,100)
    for i in lambdas:
        print(i)
        predicted_value_func = td_lambda_tabular(simulation_transitions,
                                      states,
                                      user_gamma,
                                      i,
                                      num_transitions,
                                      lr)
        list_rmses.append(get_rmse(true_value_func,predicted_value_func))
    
    
    plt.plot(lambdas,list_rmses)
    plt.xlabel("Lambda")
    plt.ylabel("RMSE")
    plt.title("Graph of convergence for different lambdas (10k transitions each time)")
    

    print("Value Function obtained by MC")
    value_mc_other = mc_prediction(
                traces = simulation_episodes,
                approx_0 = approx_0,
                γ = user_gamma
        )
    count = 0
    for episode in value_mc_other:
        count+=1
        if count%1000 == 0:
            print(f"{count} episodes processed")
        if count == 100:
            print("Value Function with Function Approximation Version")
            print(episode)
            break
        
    print("Value Function obtained by TD")
    value_td_other = td_prediction(
                transitions = simulation_transitions,
                approx_0 = approx_0,
                γ = user_gamma,
        )
    print()
    count = 0
    for i in value_td_other:
        count+=1
        if count== 100:
            print("Value Function with Function Approximation Version")
            print(i)
            break