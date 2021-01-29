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

X = TypeVar('X')
S = TypeVar('S')
A = TypeVar('A')

# A representation of a value function for a finite MDP with states of
# type S
V = Mapping[S, float]

#PROBLEM 1
#We inspired ourselves from the BSplineApprox implementation of the function_approx.py file of the RL-book repo for the direct solve part
#We made by ourselves the indirect solve part
@dataclass(frozen=True)
class BSplineApprox(FunctionApprox[X]):
    feature_function: Callable[[X], float]
    degree: int
    knots: np.ndarray = np.array([])
    #Knots should be ordered and such that the first and last node in the array specify the bounds of x we work with
    coeffs: np.ndarray = np.array([])
    direct_solve:bool = True
    num_updates: int = 0
    learning_rate: int = 0.01

    def get_feature_values(self, x_values_seq: Iterable[X]) -> Sequence[float]:
        return [self.feature_function(x) for x in x_values_seq]
    
    def get_knot(self,x):
        knot = None
        if len(self.knots)==1:
            return 0
        for i in range(len(self.knots)-1):
            if x<=self.knots[i+1]:
                return i
        if knot is None:
            return len(self.knots)-1
            
    
    def get_feature_values_degree(self, x_values_seq: Iterable[X]) -> np.ndarray:
        # We use this to compute the gradient of the loss
        return_array = np.zeros((len(x_values_seq),(self.degree+1)*(len(self.knots)-1)))
        for i in range(len(x_values_seq)):
            x = x_values_seq[i]
            knot = self.get_knot(x)
            #print(knot)
            for j in range(0,self.degree+1):
                return_array[i][knot*(self.degree+1)+j] = x**j
        return np.array(return_array)


    def representational_gradient(self, x_value: X):
        feature_val: float = self.feature_function(x_value)
        eps: float = 1e-6
        #We compute the gradient at X using Taylor formula: the difference for two really close value
        #Scipy enables us to compute this
        one_hots: np.array = np.eye(len(self.coeffs))
        return replace(
            self,
            coeffs=np.array([(
                BSpline(
                    self.knots,
                    c + one_hots[i] * eps,
                    self.degree
                )(feature_val) -
                BSpline(
                    self.knots,
                    c - one_hots[i] * eps,
                    self.degree
                )(feature_val)
            ) / (2 * eps) for i, c in enumerate(self.coeffs)]))
    
    
    def regularized_loss_gradient(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ) -> np.ndarray:
        x_vals, y_vals = zip(*xy_vals_seq)
        feature_vals: np.ndarray = self.get_feature_values_degree(x_vals) 
        #print(feature_vals)
        diff: np.ndarray = np.dot(feature_vals, self.coeffs) \
            - np.array(y_vals)
        return np.dot(feature_vals.T, diff) / len(diff)

    def evaluate(self, x_values_seq: Iterable[X]) -> np.ndarray:
        if self.direct_solve:
            spline_func: Callable[[Sequence[float]], np.ndarray] = \
                BSpline(self.knots, self.coeffs, self.degree)
            return spline_func(self.get_feature_values(x_values_seq))
        else:
            feature_vals: np.ndarray = self.get_feature_values_degree(x_values_seq)
            return np.dot(feature_vals,self.coeffs)

    def update(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]]
    ):
        if self.direct_solve:
            x_vals, y_vals = zip(*xy_vals_seq)
            feature_vals: Sequence[float] = self.get_feature_values(x_vals)
            sorted_pairs: Sequence[Tuple[float, float]] = \
                sorted(zip(feature_vals, y_vals), key=itemgetter(0))
            #We directly solve using splrep
            #We could also solve by iterated convergence using the gradients at each step and updating the coefs
            new_knots, new_coeffs, _ = splrep(
                [f for f, _ in sorted_pairs],
                [y for _, y in sorted_pairs],
                k=self.degree
            )
            return replace(
                self,
                knots=new_knots,
                coeffs=new_coeffs
            )
        else:
            #print("Update")
            if self.num_updates == 0:
                object.__setattr__(self, 'coeffs',np.zeros((self.degree+1)*(len(self.knots)-1)))
                #self.coeffs = np.zeros((self.degree+1)*len(self.knots))
                object.__setattr__(self, 'num_updates',1)
                #self.num_updates = 1
            gradient: np.ndarray = self.regularized_loss_gradient(xy_vals_seq)
            #print(gradient)
            new_coeffs = self.coeffs - self.learning_rate * gradient
            #self.coeffs = self.coeffs - self.learning_rate * gradient
            return replace(self,coeffs = new_coeffs,num_updates = 1)
            

    def solve(
        self,
        xy_vals_seq: Iterable[Tuple[X, float]],
        error_tolerance: Optional[float] = None
    ):
        if self.direct_solve:
            return self.update(xy_vals_seq)
        else:
            tol: float = 1e-3 if error_tolerance is None else error_tolerance

            def done(
                a: BSplineApprox[X],
                b: BSplineApprox[X],
                tol: float = tol
            ) -> bool:
                return a.within(b, tol)

            ret = iterate.converged(
                self.iterate_updates(itertools.repeat(xy_vals_seq)),
                done=done
            )
            return ret

    def within(self, other: FunctionApprox[X], tolerance: float) -> bool:
        #We check if the function approx other is of the same instance
        if isinstance(other, BSplineApprox):
            #print(np.abs(self.knots - other.knots))
            #print(np.abs(self.coeffs - other.coeffs))
            return \
                np.all(np.abs(self.knots - other.knots) <= tolerance).item() \
                and \
                np.all(np.abs(self.coeffs - other.coeffs) <= tolerance).item()

        return False
    
#PROBLEM 2
DEFAULT_TOLERANCE = 1e-5
#We tried two different methods 
def approximate_policy_evaluation(
        mdp: FiniteMarkovDecisionProcess[S,A],
        policy: FinitePolicy[S,A],
        vf: FunctionApprox[S],
        gamma : float) -> Iterator[FunctionApprox[S]]:
    def update(v: FunctionApprox[S]) -> Iterator[FunctionApprox[S]]:

        def return_(s_r: Tuple[S, float]) -> float:
            s1, r = s_r
            return r + gamma * v.evaluate([s1]).item()
        #print(type(v))
        return v.update(
            [(
                s,
                mdp.mapping[s][policy.policy_map[s]].expectation(return_)
            ) for s in mdp.non_terminal_states]
        )

    return iterate(update, vf)

def approximate_policy_evaluation_result(      
        mdp: FiniteMarkovDecisionProcess[S,A],
        policy: FinitePolicy[S,A],
        vf: FunctionApprox[S],
        gamma : float = 0.9):
    v_star: np.ndarray = converged(
        approximate_policy_evaluation(mdp,policy,vf,gamma),
        done=almost_equal_np_arrays
    )
    return {s: v_star[i] for i, s in enumerate(mdp.non_terminal_states)}

def greedy_policy_from_approx_vf(
    mdp: FiniteMarkovDecisionProcess[S, A],
    vf: FunctionApprox[S],
    gamma: float
) -> FinitePolicy[S, A]:
    greedy_policy_dict: Dict[S, FiniteDistribution[A]] = {}

    for s in mdp.non_terminal_states:

        q_values: Iterator[Tuple[A, float]] = \
            ((a, mdp.mapping[s][a].expectation(
                lambda s_r: s_r[1] + gamma * vf.values_map.get(s_r[0], 0.)
            )) for a in mdp.actions(s))

        greedy_policy_dict[s] =\
            Constant(max(q_values, key=operator.itemgetter(1))[0])

    return FinitePolicy(greedy_policy_dict)

def evaluate_mrp_result(
        mrp: FiniteMarkovRewardProcess[S],
        gamma: float,
        approx_0: FunctionApprox[S],
        ) -> FunctionApprox[S]:
    v_star: np.ndarray = converged(
        evaluate_finite_mrp(mrp, gamma,approx_0),
        done=almost_equal_vf_approx
    )
    return v_star


def policy_iteration(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    approx0: FunctionApprox[S]
) -> Iterator[Tuple[FunctionApprox[S], FinitePolicy[S, A]]]:
    '''Calculate the value function (V*) of the given MDP by improving
    the policy repeatedly after evaluating the value function for a policy
    '''

    def update(vf_policy: Tuple[FunctionApprox[S], FinitePolicy[S, A]])\
            -> Tuple[FunctionApprox[S], FinitePolicy[S, A]]:

        vf, pi = vf_policy
        mrp: FiniteMarkovRewardProcess[S] = mdp.apply_finite_policy(pi)
        #policy_vf: FunctionApprox[S] = approximate_policy_evaluation_result(mdp,pi,vf)
        policy_vf:  FunctionApprox[S] =  evaluate_mrp_result(mrp, gamma,vf)
        improved_pi: FinitePolicy[S, A] = greedy_policy_from_approx_vf(
            mdp,
            policy_vf,
            gamma
        )
        return policy_vf, improved_pi

    pi_0: FinitePolicy[S, A] = FinitePolicy(
        {s: Choose(set(mdp.actions(s))) for s in mdp.non_terminal_states}
    )
    return iterate(update, (approx0, pi_0))

def almost_equal_vf_approx(
    x1: FunctionApprox[S],
    x2: FunctionApprox[S]
) -> bool:
    return max(
        abs(x1.values_map[s] - x2.values_map[s]) for s in x1.values_map.keys()
    ) < DEFAULT_TOLERANCE

def almost_equal_vf_approx_pi(
    x1: Tuple[FunctionApprox[S], FinitePolicy[S, A]],
    x2: Tuple[FunctionApprox[S], FinitePolicy[S, A]]
) -> bool:
    return max(
        abs(x1[0].values_map[s] - x2[0].values_map[s]) for s in x1[0].values_map.keys()
    ) < DEFAULT_TOLERANCE


def approx_policy_iteration_result(
    mdp: FiniteMarkovDecisionProcess[S, A],
    gamma: float,
    approx0:FunctionApprox[S]
) -> Tuple[FunctionApprox[S], FinitePolicy[S, A]]:
    return converged(policy_iteration(mdp, gamma,approx0), done=almost_equal_vf_approx_pi)



    
if __name__ == '__main__':
    #PROBLEM 1
    print("Solving Problem 1")

    from scipy.stats import norm

    x_pts: Sequence[float] = np.arange(0, 3, 0.01)
    
    #Our definition of the model
    def f(x):
        if x<0 or x>3:
            raise ValueError("Value of x should be between 0 and 3")
        if x<=1:
            return -1+4*x-x**2+x**3
        elif x<=2:
            return 2*x + x**3
        elif x<=3:
            return 2-x+x**2 +x**3
        return


    n = norm(loc=0., scale=0.0001)
    
    xy_vals_seq: Sequence[Tuple[float, float]] = \
        [(x, f(x) + n.rvs(size=1)[0]) for x in x_pts]

    ff = lambda x: x
    
    BSApprox = BSplineApprox(feature_function = ff,
                             degree = 3)
    solved = BSApprox.solve(xy_vals_seq)
    errors: np.ndarray = solved.evaluate(x_pts) - np.array([y for _, y in xy_vals_seq])
    print("Mean Squared Error")
    print(np.mean(errors * errors))

    print("Indirect Solve")
    BSApprox_ind = BSplineApprox(feature_function = ff,
                             degree = 3,
                             knots = np.array([0,1,2,3]),
                             direct_solve = False)
    solved_ind = BSApprox_ind.solve(xy_vals_seq)
    errors: np.ndarray = solved_ind.evaluate(x_pts) - np.array([y for _, y in xy_vals_seq])
    print("Mean Squared Error")
    print(np.mean(errors * errors))
    #The second method takes way more time to converge
    
    #PROBLEM 2
    print("Solving Problem 2")
    n = 10
    model = LilypadModel(n)
    approx0 = Tabular(values_map = {s: 0.0 for s in model.non_terminal_states})
    result = approx_policy_iteration_result(model,0.9,approx0)
    #Results consistent with what we had before
    
