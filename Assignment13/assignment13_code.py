from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List,Optional,Iterable,TypeVar,Mapping,Dict
import numpy as np
from rl.distribution import Constant, Categorical
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian,Uniform,Constant
from rl.markov_decision_process import MarkovDecisionProcess, Policy
import rl.markov_process as mp
from rl.returns import returns
from rl.chapter2.simple_inventory_mrp import SimpleInventoryMRPFinite,InventoryState
from rl.chapter3.simple_inventory_mdp_cap import SimpleInventoryMDPCap
from rl.chapter7.asset_alloc_discrete import  AssetAllocDiscrete
from rl.markov_decision_process import FiniteMarkovDecisionProcess,policy_from_q
from rl.markov_decision_process import FinitePolicy, StateActionMapping,TransitionStep
from rl.markov_process import FiniteMarkovProcess, FiniteMarkovRewardProcess
from scipy.stats import poisson

from rl.monte_carlo import mc_prediction
from rl.td import td_prediction
from rl.function_approx import Tabular
from rl.function_approx import FunctionApprox
from rl.markov_process import FiniteMarkovRewardProcess
import matplotlib.pyplot as plt
from rl.dynamic_programming import value_iteration_result
import rl.iterate as iterate
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox
from operator import itemgetter

S = TypeVar('S')
A = TypeVar('A')

#PROBLEM 1
def mc_control_scratch(
        #traces: Iterable[Iterable[mp.TransitionStep[S]]],
        mdp_to_sample:FiniteMarkovDecisionProcess,
        states: List[S],
        actions:Mapping[S,List[A]],
        γ: float,
        tolerance: float = 1e-6,
        num_episodes:float = 10000
) -> Mapping[Tuple[S,A],float]:

    q:Mapping[Tuple[S,A],float] = {}
    counts_per_state_act:Mapping[Tuple[S,A],int] = {}
    for state in states:
        for action in actions[state]:
            q[(state,action)] = 0.
            counts_per_state_act[(state,action)] = 0
    policy_map:Mapping[S, Optional[Categorical[A]]] = {}
    for state in states:
        if actions[state] is None:
            policy_map[state] = None
        else:
            policy_map[state] = Categorical({action:1 for action in actions[state]})
    Pi:FinitePolicy[S,A] = FinitePolicy(policy_map)
    start_state_distrib = Categorical({state:1 for state in states})
    for i in range(num_episodes):
        trace:Iterable[TransitionStep[S, A]] = mdp_to_sample.simulate_actions(start_state_distrib,Pi)
        episode = returns(trace, γ, tolerance)
        #print(episode)
        for step in episode:
            state = step.state
            action = step.action
            return_ = step.return_
            counts_per_state_act[(state,action)] += 1
            q[(state,action)] += 1/counts_per_state_act[(state,action)]*(return_-q[(state,action)])
        eps = 1/(i+1)
        new_pol: Mapping[S, Optional[Categorical[A]]] = {}
        for state in states:
            if actions[state] is None:
                new_pol[state] = None
            policy_map = {action: eps/len(actions[state]) for action in actions[state]}
            best_action = actions[state][0]
            for action in actions[state]:
                if q[(state,best_action)]<=q[(state,action)]:
                    best_action = action
            policy_map[best_action] += 1-eps
            new_pol[state] = Categorical(policy_map)
        Pi = FinitePolicy(new_pol)
        
    return q


def get_opt_vf_from_q(q_value:Mapping[Tuple[S,A],float])\
    ->Tuple[Mapping[S,float],FinitePolicy[S,A]]:
        v: Mapping[S,float] = {}
        policy_map: Mapping[S, Optional[Constant[A]]] = {}
        for i in q_value:
            state,action = i
            if state not in v.keys() or q_value[i]>v[state]:
                v[state] = q_value[i]
                policy_map[state] = Constant(action)
        Pi = FinitePolicy(policy_map)
        return (v,Pi)

def mc_control_fapprox(
        mdp_to_sample: MarkovDecisionProcess[S, A],
        states: Distribution[S],
        approx_0: FunctionApprox[Tuple[S, A]],
        γ: float,
        ϵ: float,
        tolerance: float = 1e-6
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    q = approx_0
    Pi = policy_from_q(q, mdp_to_sample)

    while True:
        trace: Iterable[TransitionStep[S, A]] = mdp_to_sample.simulate_actions(states, Pi)
        q = q.update(
            ((step.state, step.action), step.return_)
            for step in returns(trace, γ, tolerance)
        )
        Pi = policy_from_q(q, mdp_to_sample, ϵ)
        yield q

#PROBLEM2
def sarsa_control_scratch(
        #traces: Iterable[Iterable[mp.TransitionStep[S]]],
        mdp_to_sample:FiniteMarkovDecisionProcess,
        states: List[S],
        actions:Mapping[S,List[A]],
        γ: float,
        num_episodes:float = 10000,
        eps:float = 0.1,
        base_lr:float = 0.03,
        half_life:float = 1000.0,
        exponent:float = 0.5
) -> Mapping[S,float]:

    q:Mapping[Tuple[S,A],float] = {}
    counts_per_state_act:Mapping[Tuple[S,A],int] = {}
    for state in states:
        for action in actions[state]:
            q[(state,action)] = 0.
            counts_per_state_act[(state,action)] = 0
    policy_map:Mapping[S, Optional[Categorical[A]]] = {}
    for state in states:
        if actions[state] is None:
            policy_map[state] = None
        else:
            policy_map[state] = Categorical({action:1 for action in actions[state]})
    Pi:FinitePolicy[S,A] = FinitePolicy(policy_map)
    state = Categorical({state:1 for state in states}).sample()
    for i in range(num_episodes):
        action_distribution = Pi.act(state)
        action = action_distribution.sample()
        next_distribution = mdp_to_sample.step(state, action)
        next_state, reward = next_distribution.sample()
        next_action = Pi.act(next_state).sample()
        counts_per_state_act[(state,action)] += 1
        alpha = base_lr / (1 + ((counts_per_state_act[(state,action)]-1)/half_life)**exponent)
        #We choose the next action based on epsilon greedy policy
        q[(state,action)] += alpha*(reward+γ*q[(next_state,next_action)]-q[(state,action)])
        new_pol: Mapping[S, Optional[Categorical[A]]] = Pi.policy_map
        if actions[state] is None:
            new_pol[state] = None
        policy_map = {action: eps/len(actions[state]) for action in actions[state]}
        best_action = actions[state][0]
        for action in actions[state]:
            if q[(state,best_action)]<=q[(state,action)]:
                best_action = action
        policy_map[best_action] += 1-eps
        new_pol[state] = Categorical(policy_map)
        Pi = FinitePolicy(new_pol)
        state = next_state
        if next_state is None:
            state = Categorical({state:1 for state in states}).sample()
    return q

def sarsa_control_fapprox(
        transitions: Iterable[TransitionStep[S, A]],
        actions: Callable[[S], Iterable[A]],
        approx_0: FunctionApprox[Tuple[S, A]],
        γ: float,
        eps:float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:
    def step(q, transition):
        if np.random.random()>eps:
            next_reward = max(
            q((transition.next_state, a))
            for a in actions(transition.next_state)
            )
        else:
            next_action = actions(transition.next_state)[np.random.randint(len(actions(transition.next_state)))]
            next_reward = ((transition.next_state, next_action))
        return q.update([
            ((transition.state, transition.action),
             transition.reward + γ * next_reward)
        ])

    return iterate.accumulate(transitions, step, initial=approx_0)

#PROBLEM 3
def qlearning_control_scratch(
        #traces: Iterable[Iterable[mp.TransitionStep[S]]],
        mdp_to_sample:FiniteMarkovDecisionProcess,
        states: List[S],
        actions:Mapping[S,List[A]],
        γ: float,
        num_episodes:float = 10000,
        eps:float = 0.1,
        base_lr:float = 0.03,
        half_life:float = 1000.0,
        exponent:float = 0.5
) -> Mapping[S,float]:

    q:Mapping[Tuple[S,A],float] = {}
    counts_per_state_act:Mapping[Tuple[S,A],int] = {}
    for state in states:
        for action in actions[state]:
            q[(state,action)] = 0.
            counts_per_state_act[(state,action)] = 0
    policy_map:Mapping[S, Optional[Categorical[A]]] = {}
    for state in states:
        if actions[state] is None:
            policy_map[state] = None
        else:
            policy_map[state] = Categorical({action:1 for action in actions[state]})
    Pi:FinitePolicy[S,A] = FinitePolicy(policy_map)
    state = Categorical({state:1 for state in states}).sample()
    for i in range(num_episodes):
        action_distribution = Pi.act(state)
        action = action_distribution.sample()
        next_distribution = mdp_to_sample.step(state, action)
        next_state, reward = next_distribution.sample()
        #We choose the next action not based on Pi that is epsilon greedy,
        #but based on the action which yields the largest q_value for next_state
        maxi:float = -np.inf
        for next_act in actions[next_state]:
            if q[(next_state,next_act)]>=maxi:
                next_action = next_act
                maxi = q[(next_state,next_act)]
        counts_per_state_act[(state,action)] += 1
        alpha = base_lr / (1 + ((counts_per_state_act[(state,action)]-1)/half_life)**exponent)
        q[(state,action)] += alpha*(reward+γ*q[(next_state,next_action)]-q[(state,action)])
        new_pol: Mapping[S, Optional[Categorical[A]]] = Pi.policy_map
        if actions[state] is None:
            new_pol[state] = None
        policy_map = {action: eps/len(actions[state]) for action in actions[state]}
        best_action = actions[state][0]
        for action in actions[state]:
            if q[(state,best_action)]<=q[(state,action)]:
                best_action = action
        policy_map[best_action] += 1-eps
        new_pol[state] = Categorical(policy_map)
        Pi = FinitePolicy(new_pol)
        state = next_state
        if next_state is None:
            state = Categorical({state:1 for state in states}).sample()
    return q

def qlearning_control_fapprox(
        transitions: Iterable[TransitionStep[S, A]],
        actions: Callable[[S], Iterable[A]],
        approx_0: FunctionApprox[Tuple[S, A]],
        γ: float
) -> Iterator[FunctionApprox[Tuple[S, A]]]:

    def step(q, transition):
        next_reward = max(
            q((transition.next_state, a))
            for a in actions(transition.next_state)
        )
        return q.update([
            ((transition.state, transition.action),
             transition.reward + γ * next_reward)
        ])

    return iterate.accumulate(transitions, step, initial=approx_0)
#Repeat code with what was done should be made easier now + test Problem 2


if __name__ == '__main__':
    print("Testing Tabular Versions")
    from pprint import pprint

    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mdp: FiniteMarkovDecisionProcess[InventoryState, int] =\
        SimpleInventoryMDPCap(
            capacity=user_capacity,
            poisson_lambda=user_poisson_lambda,
            holding_cost=user_holding_cost,
            stockout_cost=user_stockout_cost
        )
    print("MDP Value Iteration Optimal Value Function and Optimal Policy")
    print("--------------")
    opt_vf_vi, opt_policy_vi = value_iteration_result(si_mdp, gamma=user_gamma)
    pprint(opt_vf_vi)
    print(opt_policy_vi)
    print()
    dic_actions:Mapping[InventoryState,List[int]] = {}
    for i in si_mdp.mapping:
        dic_actions[i] = list(si_mdp.mapping[i].keys())
    states:List[InventoryState] = si_mdp.non_terminal_states

    print("Solving Problem 1")
    print("MC Control Algorithm")
    q_mc = mc_control_scratch(si_mdp,
                           states,
                           dic_actions,
                           user_gamma)
    opt_v_mc,opt_pol_mc = get_opt_vf_from_q(q_mc)
    print(opt_v_mc)

    print("Solving Problem 2")
    print("Sarse Control Algorithm")
    q_sarsa = sarsa_control_scratch(si_mdp,
                           states,
                           dic_actions,
                           user_gamma,
                           num_episodes = 1000000)
    opt_v_sarsa,opt_pol_sarsa = get_opt_vf_from_q(q_sarsa)
    print(opt_v_sarsa)

    print("Solving Problem 3")
    print("Tabular Q-Learning Control Algorithm")
    q_qlearning = qlearning_control_scratch(si_mdp,
                           states,
                           dic_actions,
                           user_gamma,
                           num_episodes = 1000000)
    opt_v_qlearning,opt_pol_qlearning = get_opt_vf_from_q(q_qlearning)
    print(opt_v_qlearning)

    """
    print("Testing Function Approx Versions")
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
    #THIS CODE TAKES TOO MUCH TIME TO RUN FOR US TO TEST IT
    # vf_ff: Sequence[Callable[[float], float]] = [lambda _: 1., lambda w: w]
    # it_vf: Iterator[Tuple[DNNApprox[float], Policy[float, float]]] = \
    #     aad.backward_induction_vf_and_pi(vf_ff)

    # print("Backward Induction: VF And Policy")
    # print("---------------------------------")
    # print()
    # for t, (v, p) in enumerate(it_vf):
    #     print(f"Time {t:d}")
    #     print()
    #     opt_alloc: float = p.act(init_wealth).value
    #     val: float = v.evaluate([init_wealth])[0]
    #     print(f"Opt Risky Allocation = {opt_alloc:.2f}, Opt Val = {val:.3f}")
    #     print("Weights")
    #     for w in v.weights:
    #         print(w.weights)
    #     print()

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
    Approx_0 = DNNApprox.create(feature_funcs,dnn)
    print("Monte Carlo FApprox")
    #TODO Write functioning code to test our function approx versions
    mc_fapprox = mc_control_fapprox(
        mdp_to_sample = aad.get_mdp(0),
        states = init_wealth_distr,
        approx_0 = Approx_0,
        γ = user_gamma,
        ϵ = 0.1,
        tolerance= 1e-6)
    """