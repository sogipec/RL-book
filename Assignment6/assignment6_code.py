from rl.distribution import Categorical,Constant,SampledDistribution,FiniteDistribution,Distribution,Choose
from rl.markov_process import FiniteMarkovProcess, Transition, MarkovProcess, MarkovRewardProcess,FiniteMarkovRewardProcess,StateReward,RewardTransition
from rl.markov_decision_process import FiniteMarkovDecisionProcess,FinitePolicy, StateActionMapping,ActionMapping,MarkovDecisionProcess
from rl.dynamic_programming import policy_iteration_result, value_iteration_result, almost_equal_vf_pis,almost_equal_np_arrays,greedy_policy_from_vf
from dataclasses import dataclass, replace, field
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar,NamedTuple,Callable,Iterator)
import numpy as np
import itertools
import operator
from collections import Counter
from operator import itemgetter
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func
import matplotlib.pyplot as plt
import time
from Assignment3.assignment3_code import LilypadModel
from scipy.stats import poisson
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap,InventoryState
from scipy.interpolate import splrep, BSpline
from rl.function_approx import FunctionApprox,Tabular
from rl.iterate import converged, iterate
from rl.approximate_dynamic_programming import evaluate_mrp,evaluate_finite_mrp


#PROBLEM 1

def z_opt(a:float,
          mu:float = .05,
          r:float = .03,
          sigma: float = 0.1)->float:
    num = 2*(mu-r)*(1-(1+r)*a)
    denom = 2*a*sigma**2+2*a*(mu-r)**2
    if num/denom>=1:
        return 1
    elif num/denom<0:
        return 0
    else:
        return num/denom

if __name__ == '__main__':
    print("Solving Problem 1")
    a: np.ndarray = np.linspace(0.1,1,100)
    z_values:list = []
    mu: float = .05
    r: float = .03
    sigma: float = 0.1
    for i in a:
        z_values.append(z_opt(i,mu,r,sigma))
    z_values = np.array(z_values)
    plt.plot(a,z_values)
    plt.xlabel("Value of alpha")
    plt.ylabel("Optimal z value")
    plt.title(f"mu= {mu},r = {r},sigma = {sigma}")
    plt.show()
