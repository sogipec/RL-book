from rl.distribution import Categorical,Constant,SampledDistribution,FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, Transition, MarkovProcess, MarkovRewardProcess,FiniteMarkovRewardProcess,StateReward,RewardTransition
from rl.markov_decision_process import FiniteMarkovDecisionProcess,FinitePolicy, StateActionMapping,ActionMapping
from rl.dynamic_programming import policy_iteration_result, value_iteration_result
from dataclasses import dataclass
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar,NamedTuple)
import numpy as np
import itertools
from collections import Counter
from operator import itemgetter
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
import matplotlib.pyplot as plt
import time
from Assignment3.assignment3_code import *
from scipy.stats import poisson
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap,InventoryState


#PROBLEM 2
def process_time(n,gamma = 1) -> Tuple[float,float,float]:
    print(f"n={n}")
    model = LilypadModel(n)
    start = time.time()
    list_policies = get_policies(n)
    optimal_policy,list_sum,list_values,idx_max = get_optimal_policy(n,model,list_policies,gamma = gamma) 
    time_brute = time.time()-start
    start_2 = time.time()
    value_iter = value_iteration_result(model,1)
    time_value_iter = time.time() - start_2
    start_3 = time.time()
    policy_iter = policy_iteration_result(model,1)
    time_policy_iter = time.time() - start_3
    return time_brute,time_value_iter,time_policy_iter

def plot_time(list_brute,list_value,list_policy,max_N):
    x_vals = [i for i in range(3,max_N)]
    plt.plot(x_vals,list_brute,label = "Brute Force")
    plt.plot(x_vals,list_value,label = "Value Iteration")
    plt.plot(x_vals,list_policy,label = "Policy Iteration")
    plt.xlabel("Num Lilypad")
    plt.ylabel("Convergence Time")
    plt.grid(True)
    plt.legend(loc = 'upper left')
    plt.title("Convergence Time as a function of the number of lilypads")
    return

#PROBLEM 3
class Problem3(NamedTuple):
    gamma: float
    probas: Sequence[float]
    daily_wages: Sequence[float]
    alpha: float
    tol_convergence:float
    
    def get_utility(self) -> Sequence[float]:
        list_utilities: Sequence[float] = []
        for i in self.daily_wages:
            list_utilities.append(np.log(i))
        return list_utilities
    
    def solve_value_equation(self) -> Sequence[float]:
        utilities:Sequence[float] = self.get_utility()
        epsilon = 100
        #We will work with the simplification described in the text
        #The Value function thus has n+1 dimensions instead of 2n
        value_func = [0 for i in range(len(self.probas)+1)]
        while epsilon>=self.tol_convergence:
            old_value_func = [v for v in value_func]
            int_value:float = 0.
            for i in range(len(self.probas)):
                int_value+=self.probas[i]*max(utilities[0]+self.gamma*old_value_func[0],
                                              old_value_func[i+1])
            value_func[0] = int_value
            for i in range(1,len(self.probas)+1):
                value_func[i] = utilities[i]+ self.gamma*((1-self.alpha)*old_value_func[i]+self.alpha*old_value_func[0])
            #We use the L infinite norm for the function to be a contraction
            epsilon:float = 0
            for i in range(len(old_value_func)):
                value = abs(value_func[i]-old_value_func[i])
                if value>epsilon:
                    epsilon = value
        return value_func
    
    def get_optimal_policy(self,
                           value_func:Sequence[float]) -> Sequence[str]:
        utilities:Sequence[float] = self.get_utility()
        list_actions:Sequence[str] = []
        for i in range(1,len(self.probas)+1):
            if value_func[i]> utilities[0] + self.gamma*value_func[0]:
                list_actions.append("A")
            else:
                list_actions.append("D")
        return list_actions


#PROBLEM 4
#We represent actions as a tuple of store 1 supplier order, store 2 supplier order and
#transfer between both stores

#We define a custom state class for this problem
@dataclass(frozen=True)
class State:
    state1:InventoryState
    state2:InventoryState

Problem4Mapping = StateActionMapping[State,Tuple[int,int,int]]

class ComplexMDP(FiniteMarkovDecisionProcess[State, Tuple[int,int,int]]):
    
    def __init__(
        self,
        store1: SimpleInventoryMDPCap,
        store2: SimpleInventoryMDPCap,
        K1: float,
        K2: float
        ):
        self.store1: SimpleInventoryMDPCap = store1
        self.store2: SimpleInventoryMDPCap = store2
        self.K1 = K1
        self.K2 = K2

        super().__init__(self.get_action_transition_reward_map())
    
    def get_action_transition_reward_map(self) -> Problem4Mapping:
        d: Dict[State, Dict[Tuple[int,int,int], Categorical[Tuple[State,
                                                            float]]]] = {}

        for state1 in store1.states():
            for state2 in store2.states():
                state = State(state1 = state1,state2 = state2)
                d1 : Dict[Tuple[int,int,int], Categorical[Tuple[State,
                                                                float]]] = {}
                state1_actions = store1.actions(state1)
                state2_actions = store2.actions(state2)
                list_actions: Sequence[Tuple[int,int,int]] = []
                for k in state1_actions:
                    for l in state2_actions:
                        list_actions+=[(k,l,0)]
                        #We introduce extra-actions corresponding to transfers between stores
                        for u in range(0,k):
                            if u<=state2.on_hand:
                                list_actions+=[(k-u,l,u)]
                        for u in range(0,l):
                            if u<=state1.on_hand:
                                list_actions+=[(k,l-u,-u)]
                for j in list_actions:
                    sr_probs_dict: Dict[Tuple[State, float], float] = {}
                    action1,action2,transfer = j[0],j[1],j[2]
                    #Transfers directly affect on_hand inventory
                    new_state1 = InventoryState(on_hand = state1.on_hand+transfer,on_order = state1.on_order)
                    new_state2 = InventoryState(on_hand = state2.on_hand-transfer,on_order = state2.on_order)
                    mapping1 = store1.mapping[new_state1][action1]
                    mapping2 = store2.mapping[new_state2][action2]
                    for k in mapping1:
                        for l in mapping2:

                            new_state = State(state1 = k[0][0],state2 = l[0][0])
                            reward = k[0][1]+l[0][1]
                            if action1>0:
                                reward -= self.K1
                            if action2>0:
                                reward -= self.K1
                            if transfer!=0:
                                reward -= self.K2
                            proba = k[1]*l[1]
                            sr_probs_dict[(new_state,reward)] = proba
                    try:
                        d1[j] = Categorical(sr_probs_dict)
                    except:
                        print("Error")
                        pass
                d[state] = d1
        return d


if __name__ == '__main__':
    #PROBLEM 2
    """
    print("Solving Problem 2")    
    list_brute:list = []
    list_value:list = []
    list_policy:list = []

    max_N = 16
    for i in range(3,max_N):
        print(f"Num Lilypad {i} processed")
        a,b,c = process_time(i)
        list_brute.append(a)
        list_value.append(b)
        list_policy.append(c)
    plot_time(list_brute,list_value,list_policy,max_N)

    #PROBLEM 3
    print("Solving Problem 3")    
    gamma:float = 0.8
    probas:Sequence[float] = [0.4,0.2,0.2,0.2]
    daily_wages:Sequence[float] = [1.,1.5,2.,2.5,3.]
    alpha:float = 0.1
    tol_convergence:float = 1e-6
    problem = Problem3(
        gamma = gamma,
        probas = probas,
        daily_wages = daily_wages,
        alpha = alpha,
        tol_convergence = tol_convergence)
    
    optimal_value_function = problem.solve_value_equation()
    optimal_policy = problem.get_optimal_policy(optimal_value_function)
    
    print(optimal_value_function)
    print(optimal_policy)
    """
    print("Solving Problem 4")
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0
    user_gamma = 0.9
    
    user_capacity2 = 3
    user_poisson_lambda2 = 0.9
    user_holding_cost2 = 1.5
    user_stockout_cost2 = 15.0

    store1: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        ) 
    store2: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity2,
            poisson_lambda=user_poisson_lambda2,
            holding_cost=user_holding_cost2,
            stockout_cost=user_stockout_cost2
        )
    K1 = 1
    K2 = 1
    problem4 = ComplexMDP(store1 = store1,
                          store2 = store2,
                          K1 = K1,
                          K2 = K2
                          )
    value_opt = value_iteration_result(problem4,user_gamma)
    policy_opt = policy_iteration_result(problem4,user_gamma)
    
    
    
    
    
    