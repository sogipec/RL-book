from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List,Optional,Iterable,TypeVar,Mapping,Dict
import numpy as np
from rl.distribution import Constant, Categorical
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian,Uniform,Constant,FiniteDistribution
from rl.markov_decision_process import MarkovDecisionProcess, Policy,FinitePolicy,TransitionStep
import rl.markov_process as mp
from rl.returns import returns
from rl.chapter7.asset_alloc_discrete import AssetAllocDiscrete

from rl.monte_carlo import mc_prediction
from rl.td import td_prediction
from rl.function_approx import Tabular
from rl.function_approx import FunctionApprox, LinearFunctionApprox,Weights,DNNSpec,DNNApprox
from rl.markov_process import FiniteMarkovRewardProcess
import matplotlib.pyplot as plt
from operator import itemgetter
S = TypeVar('S')
A = TypeVar('A')

#PROBLEM 1

#We will work in the finite action space setting, where we can compute the score function
#quite easily
def get_policy(theta:Sequence[float],
               features:Sequence[Callable[[S,A], float]],
               actions:Sequence[A],
               states:Sequence[S])->FinitePolicy[S,A]:
    policy_map:Mapping[S, Optional[FiniteDistribution[A]]] = {}
    for state in states:
        mapping = {}
        summ = 0
        for action in actions:
            phi:np.ndarray = np.array([feature(state,action) for feature in features])
            value = np.exp(np.dot(phi,theta))
            mapping[action] = value
            summ += value
        for action in actions:
            mapping[action] = mapping[action]/value
        policy_map[state] = Categorical(mapping)
    return FinitePolicy(policy_map)

def get_score(policy:FinitePolicy[S,A],
          state:S,
          action:A,
          features:Sequence[Callable[[S,A], float]],
          actions:Sequence[A])->Sequence[float]:
    phi_a:np.ndarray = np.array([feature(state,action) for feature in features])
    for b in actions:
        phi_b:np.ndarray = np.array([feature(state,b) for feature in features])
        pi = policy.policy_map[state].table.get(b)
        phi_b = pi*phi_b
        phi_a -= phi_b
    return list(phi_a)
    
def reinforce(mdp_to_sample_from:MarkovDecisionProcess,
              features:Sequence[Callable[[S,A], float]],
              start_state_distrib:Distribution[S],
              gamma:float,
              alpha:float,
              actions:Sequence[A],
              states:Sequence[S],
              num_episodes:int,
              max_length_episode:int)->FinitePolicy[S,A]:
    theta:np.ndarray = np.array([0 for i in range(len(features))])
    for k in range(num_episodes):
        policy = get_policy(theta,features,actions,states)
        #Not sure if we should update the policy at each step every episode or only once 
        #at the end of each episode
        steps:Iterable[TransitionStep[S, A]] = mdp_to_sample_from.simulate_actions(start_state_distrib,
                                                                                   policy)
        count = 0
        rewards:Sequence[float] = []
        state_seq:Sequence[S] = []
        action_seq:Sequence[A] = []
        for step in steps:
            count+=1
            if count>max_length_episode:
                break
            rewards.append(step.reward)
            action_seq.append(step.action)
            state_seq.append(step.state)
        G = np.zeros_like(rewards)
        for i in range(len(G)):
            somm = 0
            for j in range(i,len(G)):
                somm += gamma**(j-i)*rewards[j]
            G[i] = somm
        for i in range(len(G)):
            s = state_seq[i]
            a = action_seq[i]
            score = get_score(policy,s,a,features,actions)
            theta += alpha*(gamma**i)*score*G[i]
    return policy
    
#Problem 2
#We implement the ACTOR-CRITIC-ELIGIBILITY-TRACES Algorithm in a finite state setting
#We will approximate the value function using a Linear Function 

def actor_critic_et(mdp_to_sample_from:MarkovDecisionProcess,
              features:Sequence[Callable[[S,A], float]],
              start_state_distrib:Distribution[S],
              gamma:float,
              alpha_v:float,
              alpha_theta:float,
              lambda_v:float,
              lambda_theta:float,
              actions:Sequence[A],
              states:Sequence[S],
              num_episodes:int,
              max_length_episode:int,
              approx_0:LinearFunctionApprox[S])->FinitePolicy[S,A]:
    theta:np.ndarray = ([0 for i in range(len(features))])
    V:LinearFunctionApprox[S] = approx_0
    v = V.weights.weights
    feature_functions_v = V.feature_functions
    regularization_coeff_v = V.regularization_coeff
    for k in range(num_episodes):
        policy = get_policy(theta,features,actions,states)
        #Same here, don't know if we should update the policy at the end of each episode or no
        steps:Iterable[TransitionStep[S, A]] = mdp_to_sample_from.simulate_actions(start_state_distrib,
                                                                                   policy)
        P = 1
        z_theta,z_v = np.zeros(len(features)),np.zeros(len(features))
        count = 0
        for step in steps:
            #policy = get_policy(theta,features,actions,states)
            V = LinearFunctionApprox(feature_functions = feature_functions_v,
                                 regularization_coeff = regularization_coeff_v,
                                 weights = Weights.create(weights=v),
                                 direct_solve = False)
            #Don't know if we should update V at the end of each episode or no
            count+=1
            if count>max_length_episode:
                break
            action = step.action
            state = step.state
            next_state = step.next_state
            reward = step.reward
            v_s,v_nexts = V.evaluate([state,next_state])
            delta = reward + gamma*v_nexts - v_s
            z_v = gamma*lambda_v*z_v + V.representational_gradient(state)
            z_theta = gamma*lambda_theta*z_theta + P*get_score(policy,state,action,features,actions)
            theta = theta+alpha_theta*delta*z_theta
            v = v + alpha_v*delta*z_v
            P = gamma*P
    return policy
            
            
if __name__ == '__main__':

    from pprint import pprint

    steps: int = 4
    μ: float = 0.13
    σ: float = 0.2
    r: float = 0.07
    a: float = 1.0
    init_wealth: float = 1.0
    init_wealth_var: float = 0.1

    excess: float = μ - r
    var: float = σ * σ
    base_alloc: float = excess / (a * var)

    risky_ret: Sequence[Gaussian] = [Gaussian(μ=μ, σ=σ) for _ in range(steps)]
    riskless_ret: Sequence[float] = [r for _ in range(steps)]
    utility_function: Callable[[float], float] = lambda x: - np.exp(-a * x) / a
    alloc_choices: Sequence[float] = np.linspace(
        2 / 3 * base_alloc,
        4 / 3 * base_alloc,
        11
    )
    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0],
            lambda w_x: w_x[1],
            lambda w_x: w_x[1] * w_x[1]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.sign(a) * np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    init_wealth_distr: Gaussian = Gaussian(μ=init_wealth, σ=init_wealth_var)

    aad: AssetAllocDiscrete = AssetAllocDiscrete(
        risky_return_distributions=risky_ret,
        riskless_returns=riskless_ret,
        utility_func=utility_function,
        risky_alloc_choices=alloc_choices,
        feature_functions=feature_funcs,
        dnn_spec=dnn,
        initial_wealth_distribution=init_wealth_distr
    )
    #The line of code below takes too much time to run, we can't compare the results of our implementation
    #with the baseline
    """
    it_qvf: Iterator[DNNApprox[Tuple[float, float]]] = \
        aad.backward_induction_qvf()

    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_alloc: float = max(
            ((q.evaluate([(init_wealth, ac)])[0], ac) for ac in alloc_choices),
            key=itemgetter(0)
        )[1]
        val: float = max(q.evaluate([(init_wealth, ac)])[0]
                         for ac in alloc_choices)
        print(f"Opt Risky Allocation = {opt_alloc:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
        print()

    print("Analytical Solution")
    print("-------------------")
    print()

    for t in range(steps):
        print(f"Time {t:d}")
        print()
        left: int = steps - t
        growth: float = (1 + r) ** (left - 1)
        alloc: float = base_alloc / growth
        val: float = - np.exp(- excess * excess * left / (2 * var)
                              - a * growth * (1 + r) * init_wealth) / a
        bias_wt: float = excess * excess * (left - 1) / (2 * var) + \
            np.log(np.abs(a))
        w_t_wt: float = a * growth * (1 + r)
        x_t_wt: float = a * excess * growth
        x_t2_wt: float = - var * (a * growth) ** 2 / 2

        print(f"Opt Risky Allocation = {alloc:.3f}, Opt Val = {val:.3f}")
        print(f"Bias Weight = {bias_wt:.3f}")
        print(f"W_t Weight = {w_t_wt:.3f}")
        print(f"x_t Weight = {x_t_wt:.3f}")
        print(f"x_t^2 Weight = {x_t2_wt:.3f}")
        print()
      
    """
        