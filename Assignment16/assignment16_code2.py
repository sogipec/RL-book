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

def reinforce(mdp_to_sample_from:MarkovDecisionProcess,
              features:Sequence[Callable[[S,A], float]],
              start_state_distrib:Distribution[S],
              gamma:float,
              alpha:float,
              actions:Callable[[S], Iterable[A]],
              num_episodes:int,
              max_length_episode:int)->Policy[S,A]:
    
    def get_phi(state:S,action:A)->np.ndarray:
        return np.array([feature((state,action)) for feature in features])
    
    class SoftPolicy(Policy[S,A]):
        def __init__(self,theta:np.ndarray):
            self.theta = theta
        def act(self,state:S)->Optional[Distribution[A]]:
            mapping = {}
            denom = np.sum([np.exp(np.dot(get_phi(state, b), self.theta)) for b in actions(state)
                                      if np.exp(np.dot(get_phi(state, b), self.theta)) >= 0.0001])
            for action in actions(state):
                phi:np.ndarray = get_phi(state,action)
                value = np.exp(np.dot(phi,theta))
                if value<0.00001:
                    continue
                mapping[action] = value/denom
            return Categorical(mapping)
        
    theta:np.ndarray = np.zeros(len(features))  
    for k in range(num_episodes):
        print(k)
        policy = SoftPolicy(theta)
        #Not sure if we should update the policy at each step every episode or only once 
        #at the end of each episode
        steps:Iterable[TransitionStep[S, A]] = mdp_to_sample_from.simulate_actions(start_state_distrib,
                                                                                   policy)
        episode = list(returns(steps,gamma,gamma**30))
        for k in range(len(episode)):
            step = episode[k]
            s = step.state
            a = step.action
            return_ = step.return_
            phi_a:np.ndarray = get_phi(s,a)
            denom = sum([np.exp(np.dot(get_phi(s, b), theta)) for b in actions(s)
                                if np.exp(np.dot(get_phi(s, b), theta)) >= 0.0001])
            pi = sum([np.exp(np.dot(get_phi(s, b), theta)) * get_phi(s, b) for b in actions(s)
                          if np.exp(np.dot(get_phi(s, b), theta)) >= 0.0001])
            if denom == 0:
                subtract = 0
            else:
                subtract = pi/denom
            score = phi_a - subtract
            theta += alpha*(gamma**k)*score*return_
    policy = SoftPolicy(theta)
    return policy
    
#PROBLEM 2
#We implement the ACTOR-CRITIC-ELIGIBILITY-TRACES Algorithm
def actor_critic_et(mdp_to_sample_from:MarkovDecisionProcess,
              features:Sequence[Callable[[S,A], float]],
              start_state_distrib:Distribution[S],
              gamma:float,
              alpha_v:float,
              alpha_theta:float,
              lambda_v:float,
              lambda_theta:float,
              actions:Callable[[S], Iterable[A]],
              num_episodes:int,
              max_length_episode:int,
              approx_0:DNNApprox[S])->Policy[S,A]:
    
    def get_phi(state:S,action:A)->np.ndarray:
        return np.array([feature((state,action)) for feature in features])
    
    class SoftPolicy(Policy[S,A]):
        def __init__(self,theta:np.ndarray):
            self.theta = theta
        def act(self,state:S)->Optional[Distribution[A]]:
            mapping = {}
            denom = np.sum([np.exp(np.dot(get_phi(state, b), self.theta)) for b in actions(state)
                                      if np.exp(np.dot(get_phi(state, b), self.theta)) >= 0.0001])
            for action in actions(state):
                phi:np.ndarray = get_phi(state,action)
                value = np.exp(np.dot(phi,theta))
                if value<0.00001:
                    continue
                mapping[action] = value/denom
            return Categorical(mapping)
        
    theta:np.ndarray = np.zeros(len(features))
    V:DNNApprox[S] = approx_0
    v = [V.weights[i].weights for i in range(len(V.weights))]
    feature_functions_v = V.feature_functions
    regularization_coeff_v = V.regularization_coeff
    dnn_spec = V.dnn_spec
    for k in range(num_episodes):
        policy = SoftPolicy(theta)
        #Same here, don't know if we should update the policy at the end of each episode or no
        steps:Iterable[TransitionStep[S, A]] = mdp_to_sample_from.simulate_actions(start_state_distrib,
                                                                                   policy)
        P = 1
        z_theta,z_v = np.zeros(len(features)),np.zeros(len(feature_functions_v))
        count = 0
        for step in steps:
            #policy = get_policy(theta,features,actions,states)
            V = DNNApprox(feature_functions = feature_functions_v,
                          dnn_spec= dnn_spec,
                          weights = [Weights.create(weights=i) for i in v],
                          regularization_coeff = regularization_coeff_v)
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
            phi_a:np.ndarray = get_phi(state,action)
            denom = sum([np.exp(np.dot(get_phi(state, b), theta)) for b in actions(state)
                                if np.exp(np.dot(get_phi(state, b), theta)) >= 0.0001])
            pi = sum([np.exp(np.dot(get_phi(state, b), theta)) * get_phi(state, b) for b in actions(state)
                          if np.exp(np.dot(get_phi(state, b), theta)) >= 0.0001])
            if denom == 0:
                subtract = 0
            else:
                subtract = pi/denom
            score = phi_a - subtract
            z_theta = gamma*lambda_theta*z_theta + P*score
            theta = theta+alpha_theta*delta*z_theta
            v = v + alpha_v*delta*z_v
            P = gamma*P
    policy = SoftPolicy(theta)
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

    mdp = aad.get_mdp(3)
    num_episodes = 100
    max_length_episode = 10000
    print("Solving Problem 1")

    policy = reinforce(mdp,
                       feature_funcs,
                       init_wealth_distr,
                       0.95,
                       0.01,
                       mdp.actions,
                       num_episodes,
                       max_length_episode)
    """
    
    print("Solving Problem 2")
    ffs =  [
            lambda _: 1.,
            lambda w_x: w_x,
            lambda w_x: w_x**2
        ]
    approx_0 = aad.get_vf_func_approx(ffs)
    #Some issues with function approx and weights
    policy_2 = actor_critic_et(mdp,
                       feature_funcs,
                       init_wealth_distr,
                       0.95,
                       0.01,0.01,0.01,0.01,
                       mdp.actions,
                       num_episodes,
                       max_length_episode,
                       approx_0)
    """
                                

        