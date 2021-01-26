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
    """
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
    
    
    
    
    