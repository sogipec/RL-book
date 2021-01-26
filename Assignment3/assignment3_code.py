from rl.distribution import Categorical,Constant,SampledDistribution,FiniteDistribution
from rl.markov_process import FiniteMarkovProcess, Transition, MarkovProcess, MarkovRewardProcess,FiniteMarkovRewardProcess,StateReward,RewardTransition
from rl.markov_decision_process import FiniteMarkovDecisionProcess,FinitePolicy, StateActionMapping,ActionMapping
from dataclasses import dataclass
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar)
import numpy as np
import itertools
from collections import Counter
from operator import itemgetter
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class StatePond:
    position: int
    
@dataclass(frozen=True)    
class Action:
    action: str
    
class LilypadModel(FiniteMarkovDecisionProcess[StatePond,Action]):
        def __init__(
            self,
            n:int):
            self.n:int = n
            #The grid is represented as a list of 
            super().__init__(self.get_lilypads_mapping())

        def get_lilypads_mapping(self) -> StateActionMapping[StatePond, Action]:
            n:int = self.n
            mapping: StateActionMapping[StatePond, Action] = {}
            for i in range(1,n):
                mapping[StatePond(position = i)] = {
                        Action('A'): Categorical({
                            (StatePond(position = i-1),0.): i / n,
                            (StatePond(position = i+1),1. if i==n-1 else 0.): (1. - i / n)
                        }),
                        Action('B'): Categorical({
                            (StatePond(position = j),1. if j==n else 0.): (1 / n)
                            for j in range(n + 1) if j != i
                        })
                    } 
            mapping[StatePond(position = 0)] = None
            mapping[StatePond(position = n)] = None
            return mapping

    
def get_policies(n)->Iterable[FinitePolicy[StatePond,Action]]: 
    list_policies: Iterable[FinitePolicy[StatePond,Action]] = []
    liste_actions:list = list(itertools.product(['A','B'],repeat=n-1))
    for i in liste_actions:
        policy_map: Mapping[StatePond, Optional[FiniteDistribution[Action]]] = {}
        policy_map[StatePond(0)] = None
        policy_map[StatePond(n)] = None
        for j in range(0,n-1):
            policy_map[StatePond(j+1)] = Constant(Action(i[j]))
        list_policies+=[FinitePolicy(policy_map)]
    return list_policies
            
def get_optimal_policy(n,model,list_policies,gamma) -> FinitePolicy[StatePond,Action] :
    list_sum = []
    list_values = []
    #The policy with the biggest associated sum of value functions dominates all the others
    for i in list_policies:
        markov_reward = model.apply_finite_policy(i)
        value_vec = markov_reward.get_value_function_vec(gamma = gamma)
        list_sum += [np.sum(value_vec)] 
        list_values+=[value_vec]
    list_sum = np.array(list_sum)
    list_values = np.array(list_values)
    idx_max = np.argsort(-list_sum)[0]
    return list_policies[idx_max],list_sum,list_values,idx_max

def plots(n,optimal_policy,list_values_optimal):
    x_vals = [i for i in range(1,n)]
    optimal_actions = []
    for i in optimal_policy.policy_map.keys():
        if optimal_policy.policy_map[i] is not None:
            optimal_actions+=[optimal_policy.policy_map[i].value.action]
        #optimal_actions += [optimal_policy.policy_map[StatePond(position=i)].value.action]
    plt.plot(x_vals,optimal_actions)
    plt.xlabel("Lilypad")
    plt.ylabel("Optimal Actions")
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.title(f"n={n}")
    plt.show()
    plt.plot(x_vals,list_values_optimal)
    plt.ylim(ymin=0, ymax=1)
    plt.xlabel("Lilypad")
    plt.ylabel("Optimal Escape Prob")
    plt.grid(True)
    plt.legend(loc = 'lower right')
    plt.title(f"n={n}")
    plt.show()
    return
    
def process(n,gamma = 1):
    print(f"n={n}")
    model = LilypadModel(n)
    list_policies = get_policies(n)
    optimal_policy,list_sum,list_values,idx_max = get_optimal_policy(n,model,list_policies,gamma = gamma) 
    plots(n,optimal_policy,list_values[idx_max])
    print(f"The optimal policy is: \n {optimal_policy}")
    print(f"The optimal value function is: \n {list_values[idx_max]}")
    return


    
if __name__ == '__main__':
    process(3)
    process(6)
    process(9)








