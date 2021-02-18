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
from rl.markov_process import FiniteMarkovRewardProcess

S = TypeVar('S')
A = TypeVar('A')

#Problem 1
def mc_prediction_scratch(
        traces: Iterable[Iterable[mp.TransitionStep[S]]],
        states: List[S],
        γ: float,
        tolerance: float = 1e-6,
        num_episodes:float = 10000
) -> Mapping[S,float]:
    
    '''
    Evaluate an MRP using the monte carlo method, simulating episodes
    of the given number of steps.

    Each value this function yields represents the approximated value
    function for the MRP after one additional epsiode.

    Arguments:
      traces -- an iterator of simulation traces from an MRP
      states -- list of all possible states
      γ -- discount rate (0 < γ ≤ 1), default: 1
      tolerance -- a small value—we stop iterating once γᵏ ≤ tolerance
    Returns a 

    '''
    v:Mapping[S,float] = {}
    counts_per_state:Mapping[S,int] = {}
    for state in states:
        v[state] = 0.
        counts_per_state[state] = 0
    episodes = (returns(trace, γ, tolerance) for trace in traces)
    count_episodes = 0
    for episode in episodes:
        count_episodes += 1
        if count_episodes>num_episodes:
            break
        if count_episodes%1000 == 0:
            print(f"{count_episodes} episodes processed")
        for step in episode:
            count:int = counts_per_state[state]
            v[step.state] = v[step.state]*(count/(count+1))+1/(count+1)*step.return_
            counts_per_state[state] = count + 1

    return v

#Problem 2
def td_prediction_scratch(
        transitions: Iterable[mp.TransitionStep[S]],
        states:List[S],
        γ: float,
        num_transitions: float = 10000,
        learning_rate:float = 0.1
) -> Mapping[S,float]:
    '''Evaluate an MRP using TD(0) using the given sequence of
    transitions.

    Each value this function yields represents the approximated value
    function for the MRP after an additional transition.

    Arguments:
      transitions -- a sequence of transitions from an MRP which don't
                     have to be in order or from the same simulation
      states -- list of all possible states
      γ -- discount rate (0 < γ ≤ 1)

    '''
    v:Mapping[S,float] = {}
    counts_per_state:Mapping[S,int] = {}
    for state in states:
        v[state] = 0.
        counts_per_state[state] = 0
    count_transitions = 0
    for transition in transitions:
        count_transitions+=1
        if count_transitions>num_transitions:
            break
        counts_per_state[state] += 1
        #learning_rate = 1/counts_per_state[state]
        #learning_rate = 0.1
        v[transition.state] = v[transition.state] + learning_rate*\
            (transition.reward + γ * v[transition.next_state] - v[transition.state])
    return v

#Problem 4
class RandomWalkMRP2D(FiniteMarkovRewardProcess[Tuple[int,int]]):
    '''
    2D representation of RandomWalkMRP
    '''
    barrier_x: int
    barrier_y:int
    p_up: float
    p_down: float
    p_left:float

    def __init__(
        self,
        barrier_x: int,
        barrier_y:int,
        p_up: float,
        p_down: float,
        p_left:float,
    ):
        self.barrier_x = barrier_x
        self.barrier_y = barrier_y
        self.p_up = p_up
        self.p_down = p_down
        self.p_left = p_left
        super().__init__(self.get_transition_map())

    def get_transition_map(self) -> \
            Mapping[int, Optional[Categorical[Tuple[int, float]]]]:
        d: Dict[int, Optional[Categorical[Tuple[int, float]]]] = {}
        for x in range(self.barrier_x):
            d[(x,0)] = None
            d[(x,self.barrier_y)] = None
        for y in range(self.barrier_y):
            d[(0,y)] = None
            d[(self.barrier_x,y)] = None
        for x in range(1,self.barrier_x):
            for y in range(1,self.barrier_y):
                d[(x,y)] = Categorical({
                ((x + 1, y),0. if x+1 < self.barrier_x - 1 else 1.): 1- self.p_up-self.p_down-self.p_left,
                ((x - 1 , y),0. ): self.p_left,
                ((x , y+1),0. if y+1 < self.barrier_y - 1 else 1.): self.p_up,
                ((x  , y-1),0.): self.p_left,
                })
        return d
    
if __name__ == '__main__':

    print("Testing our implementations for Problems 1 and 2 and solving problem 3")
    user_capacity = 2
    user_poisson_lambda = 1.0
    user_holding_cost = 1.0
    user_stockout_cost = 10.0

    user_gamma = 0.9

    si_mrp = SimpleInventoryMRPFinite(
        capacity=user_capacity,
        poisson_lambda=user_poisson_lambda,
        holding_cost=user_holding_cost,
        stockout_cost=user_stockout_cost
    )
    print("Value Function")
    print("--------------")
    si_mrp.display_value_function(gamma=user_gamma)
    print()
    
    states:List[InventoryState] = si_mrp.non_terminal_states
    start_state_distrib: Categorical[InventoryState] = Categorical({i:1 for i in states})
    simulation_episodes = si_mrp.reward_traces(start_state_distrib)
    simulation_transitions = si_mrp.simulate_reward(start_state_distrib)
    approx_0 = Tabular({i : 0 for i in states})
    value_mc = mc_prediction_scratch(
                traces = simulation_episodes,
                states = states,
                γ = user_gamma,
                tolerance = 1e-6,
                num_episodes = 10000
        )
    print("Value Function with our implementation of MC")
    print(value_mc)
    
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
        if count ==10000:
            print("Value Function with Function Approximation Version")
            print(episode)
            break
    
    value_td = td_prediction_scratch(
        transitions = simulation_transitions,
        states = states,
        γ = user_gamma,
        num_transitions = 100000,
        learning_rate = 0.1
        ) 
    print("Value Function with our implementation of TD")
    print(value_td)
    

    value_td_other = td_prediction(
                transitions = simulation_transitions,
                approx_0 = approx_0,
                γ = user_gamma,
        )
    count = 0
    for i in value_td_other:
        count+=1
        if count==100000:
            print("Value Function with Function Approximation Version")
            print(i)
            break

    print("Solving Problem 4")
    from rl.chapter10.prediction_utils import compare_td_and_mc

    this_barrier_x: int = 10
    this_barrier_y: int = 10
    p_up: float = 0.25
    p_down: float = 0.25
    p_left: float = 0.25
    random_walk: RandomWalkMRP2D = RandomWalkMRP2D(
        barrier_x=this_barrier_x,
        barrier_y=this_barrier_y,
        p_up=p_up,
        p_down = p_down,
        p_left = p_left
    )
    compare_td_and_mc(
        fmrp=random_walk,
        gamma=1.0,
        mc_episode_length_tol=1e-6,
        num_episodes=700,
        learning_rates=[(0.01, 1e8, 0.5), (0.05, 1e8, 0.5)],
        initial_vf_dict={s: 0.5 for s in random_walk.non_terminal_states},
        plot_batch=7,
        plot_start=0
    )
    
    

    
        