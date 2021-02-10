from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List
import numpy as np
from rl.dynamic_programming import V
from scipy.stats import norm
from rl.markov_decision_process import FinitePolicy
from rl.distribution import Constant, Categorical
from rl.finite_horizon import optimal_vf_and_policy
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox,LinearFunctionApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy,value_iteration
from rl.approximate_dynamic_programming import back_opt_qvf
from operator import itemgetter

#PROBLEM 3

#We tried two different solutions to model this problem as a MDP
#The first one involves considering lots of different MDPs, one for each time step with state (Asset_Price_t)
#The second one involves considering one single MDP with state (time,Asset_Price at t)

@dataclass(frozen=True)
class AmericanOption():
    
    asset_price_distribution: SampledDistribution[float]
    payoff: Callable[[float], float]
    expiry: int
    feature_functions: Sequence[Callable[[Tuple[float, float]], float]]
    dnn_spec: DNNSpec
    
    def time_steps(self) -> int:
        return self.expiry
    
    def get_mdp_v2(self) -> MarkovDecisionProcess[float, float]:
            """
            We tried to represent thi
            State is (t,Asset Price), Action is selling or not (= x_t)
            """
            expiry_time = self.expiry
            payoffs = self.payoff
            asset_distribution  = self.asset_price_distribution
            class AmericanMDP(MarkovDecisionProcess[Tuple[int,float], bool]):
    
                def step(
                    self,
                    state: Tuple[int,float],
                    action: bool
                ) -> SampledDistribution[Tuple[int, float]]:
                    if state[0] > expiry_time or state[0] == -1:
                        return None
                    elif action:
                        return Constant(((-1,state[1]),payoffs(state[1])))
                    else:
                        def sr_sampler_func(
                            state = state,
                            action = action
                        ) -> Tuple[Tuple[int,float], float]:
                            next_state_price: float = asset_distribution.sample()
                            next_state_time = state[0]+1
                            reward: float = 0
                            return ((next_state_time,next_state_price), reward)
        
                        return SampledDistribution(
                            sampler=sr_sampler_func,
                            expectation_samples=1000
                        )
    
                def actions(self, state: Tuple[int,float]) -> Sequence[bool]:
                    time_step = state[0]
                    if time_step>expiry_time or time_step == -1:
                        return [False]
                    else:
                        return [True,False]
    
            return AmericanMDP()
        
    def get_mdp(self,t:int) -> MarkovDecisionProcess[float, float]:
        """
        State is Asset Price_t, Action is selling or not (= x_t)
        """
        expiry_time = self.expiry
        payoffs = self.payoff
        asset_distribution  = self.asset_price_distribution
        class AmericanMDP(MarkovDecisionProcess[float, bool]):

            def step(
                self,
                state: float,
                action: bool
            ) -> SampledDistribution[Tuple[float, float]]:
                if action:
                    return Constant((state,payoffs(state)))
                else:
                    def sr_sampler_func(
                        state = state,
                        action = action
                    ) -> Tuple[float, float]:
                        next_state_price: float = asset_distribution.sample()
                        reward: float = 0
                        return (next_state_price, reward)
    
                    return SampledDistribution(
                        sampler=sr_sampler_func,
                        expectation_samples=1000
                    )

            def actions(self, state: float) -> Sequence[bool]:
                if t > expiry_time or t == -1:
                    return [False]
                else:
                    return [True,False]
    
        return AmericanMDP()
    def get_qvf_func_approx(self) -> DNNApprox[Tuple[float, float]]:

        adam_gradient: AdamGradient = AdamGradient(
            learning_rate=0.1,
            decay1=0.9,
            decay2=0.999
        )
        return DNNApprox.create(
            feature_functions=self.feature_functions,
            dnn_spec=self.dnn_spec,
            adam_gradient=adam_gradient
        )

    def get_states_distribution(self, t: int) -> SampledDistribution[float]:
        return self.asset_price_distribution

    def backward_induction_qvf(self) -> \
            Iterator[DNNApprox[Tuple[float, float]]]:

        init_fa: DNNApprox[Tuple[float, float]] = self.get_qvf_func_approx()

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[float, float],
            DNNApprox[Tuple[float, float]],
            SampledDistribution[float]
        ]] = [(
            self.get_mdp(i),
            init_fa,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps())]

        num_state_samples: int = 100
        error_tolerance: float = 1e-2

        return back_opt_qvf(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=1.0,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )






if __name__ == '__main__':
    print("Solving Problem 3")
    from pprint import pprint
    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: int = 3
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 300
    sigma: float = 1
    asset_price_distribution: SampledDistribution[float] = Gaussian(strike, sigma)

    if is_call:
        opt_payoff = lambda  x: max(x - strike, 0)
    else:
        opt_payoff = lambda  x: max(strike - x, 0)
    

    feature_funcs: Sequence[Callable[[Tuple[float, float]], float]] = \
        [
            lambda _: 1.,
            lambda w_x: w_x[0]
        ]
    dnn: DNNSpec = DNNSpec(
        neurons=[],
        bias=False,
        hidden_activation=lambda x: x,
        hidden_activation_deriv=lambda y: np.ones_like(y),
        output_activation=lambda x: - np.exp(-x),
        output_activation_deriv=lambda y: -y
    )
    american_option: AmericanOption = AmericanOption(
        asset_price_distribution = asset_price_distribution,
        payoff = opt_payoff,
        expiry = expiry_val,
        dnn_spec = dnn,
        feature_functions = feature_funcs)
    

    print("Using Method 1")
    #This does not work all the time, it works mostly when executing the line below separately from the rest
    it_qvf: Iterator[DNNApprox[Tuple[float, float]]] = \
        american_option.backward_induction_qvf()
    print("Backward Induction on Q-Value Function")
    print("--------------------------------------")
    print()
    actions = [True,False]
    #Initial Price = 100
    for t, q in enumerate(it_qvf):
        print(f"Time {t:d}")
        print()
        opt_act: float = max(
            ((q.evaluate([(strike, ac)])[0], ac) for ac in actions),
            key=itemgetter(0)
        )[1]
        val: float = max(q.evaluate([(strike, ac)])[0]
                         for ac in actions)
        print(f"Opt Action = {opt_act:.3f}, Opt Val = {val:.3f}")
        print("Optimal Weights below:")
        for wts in q.weights:
            pprint(wts.weights)
        print()

    print("Using Method 2")
    mdp = american_option.get_mdp_v2()
    
    class InitialDistrib(SampledDistribution[Tuple[int,float]]):
        '''A Gaussian distribution with the given μ and σ.'''
    
        μ: float
        σ: float
        expiry: int
    
        def __init__(self, μ: float, σ: float,expiry: int, expectation_samples: int = 10000):
            self.μ = μ
            self.σ = σ
            super().__init__(
                sampler=lambda: (np.random.randint(expiry+1),np.random.normal(loc=self.μ, scale=self.σ)),
                expectation_samples=expectation_samples
            )
    
    nt_states_distribution = InitialDistrib(strike,sigma,expiry_val)
    ag = AdamGradient(
        learning_rate=0.5,
        decay1=0.9,
        decay2=0.999
    )
    ffs = [
        lambda x: x[0],
        lambda x: x[1]
    ]

    lfa = LinearFunctionApprox.create(
         feature_functions=ffs,
         adam_gradient=ag,
         regularization_coeff=0.001,
         direct_solve=True
    )   
    solution_2 = value_iteration(mdp,
                                 1,
                                 lfa,
                                 nt_states_distribution,
                                 100)
    """
    for i in solution_2:
        print(i)
    """
    #This second method does not really work
        

        


