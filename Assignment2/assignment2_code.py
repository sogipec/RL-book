from rl.distribution import Categorical,Constant,SampledDistribution
from rl.markov_process import FiniteMarkovProcess, Transition, MarkovProcess, MarkovRewardProcess,FiniteMarkovRewardProcess,StateReward,RewardTransition
from dataclasses import dataclass
from typing import (Dict, Iterable, Generic, Sequence, Tuple,
                    Mapping, Optional, TypeVar)
import numpy as np
import itertools
from collections import Counter
from operator import itemgetter
from rl.gen_utils.plot_funcs import plot_list_of_curves
from rl.gen_utils.common_funcs import get_logistic_func, get_unit_sigmoid_func

################################################################################
################## Classes and Functions for Questions 1/2 #####################
################################################################################

#The class below represents Snakes And Ladders in the game
#Snakes will have a start superior to the end and ladders will have an end superior to the starts
@dataclass
class SnakesAndLadders:
    start:int
    end:int
#This is our class for the state in the Snakes and Ladders game
@dataclass(frozen=True)
class StateSnakeAndLadder:
    position:int

class SnakesAndLaddersGame(FiniteMarkovProcess[StateSnakeAndLadder]):
    def __init__(
            self,
            grid_size:int,
            grid:Iterable[SnakesAndLadders]):
        self.grid_size:int = grid_size
        #The grid is represented as a list of 
        self.grid: Iterable[SnakesAndLadders] = grid
        super().__init__(self.get_transition_map())
        
    def get_transition_map(self) -> Transition[StateSnakeAndLadder]:
        d: Dict[StateSnakeAndLadder, Categorical[StateSnakeAndLadder]] = {}
        dic_mapping = {}
        for i in self.grid:
            dic_mapping[i.start] = i.end
        for i in range(1,self.grid_size):
            state = StateSnakeAndLadder(position = i)
            dic_positions_associated : dict = {}
            for j in range(i+1,i+7):
                if(dic_mapping.get(j)) is not None:
                    new_pos = dic_mapping[j]
                else:
                    new_pos = j
                if new_pos>self.grid_size:
                    new_pos = self.grid_size
                if new_pos in dic_positions_associated.keys():
                    dic_positions_associated[new_pos] += 1
                else:
                    dic_positions_associated[new_pos] = 1
            state_probs_map: Mapping[StateSnakeAndLadder,float] = {
                StateSnakeAndLadder(position = j): dic_positions_associated[j]/6 for j in dic_positions_associated.keys()}
            d[state] = Categorical(state_probs_map)
        d[StateSnakeAndLadder(position = self.grid_size)] = None
        return d


def process_traces(time_steps: int,
                   num_traces: int,
                   game: SnakesAndLaddersGame) -> np.array:
    start_state_distribution = Constant(StateSnakeAndLadder(position = 1))
    array_length = []
    for i in range(num_traces):
        new_val = np.fromiter((s.position for s in itertools.islice(game.simulate(start_state_distribution),time_steps + 1)),float)
        array_length += [len(new_val)]
    return np.array(array_length)

def get_terminal_histogram(
    price_traces: np.array
) -> Tuple[Sequence[int], Sequence[int]]:
    pairs = sorted(
        list(Counter(price_traces).items()),
        key=itemgetter(0)
    )
    return [x for x, _ in pairs], [y for _, y in pairs]

################################################################################
################## Classes and Functions for Question 4#########################
################################################################################
class SnakesAndLaddersRewards(FiniteMarkovRewardProcess[StateSnakeAndLadder]):
    def __init__(
            self,
            grid_size:int,
            grid:Iterable[SnakesAndLadders]):
        self.grid_size:int = grid_size
        #The grid is represented as a list of 
        self.grid: Iterable[SnakesAndLadders] = grid
        super().__init__(self.get_transition_reward_map())


    def get_transition_reward_map(self) -> RewardTransition[StateSnakeAndLadder]:
        d: Dict[StateSnakeAndLadder, Categorical[Tuple[StateSnakeAndLadder, float]]] = {}
        dic_mapping = {}
        for i in self.grid:
            dic_mapping[i.start] = i.end
        for i in range(1,self.grid_size):
            state = StateSnakeAndLadder(position = i)
            dic_positions_associated : dict = {}
            for j in range(i+1,i+7):
                if(dic_mapping.get(j)) is not None:
                    new_pos = dic_mapping[j]
                else:
                    new_pos = j
                if new_pos>self.grid_size:
                    new_pos = self.grid_size
                if new_pos in dic_positions_associated.keys():
                    dic_positions_associated[new_pos] += 1
                else:
                    dic_positions_associated[new_pos] = 1
            sr_probs_map: Dict[Tuple[StateSnakeAndLadder, float], float] = \
                {(StateSnakeAndLadder(position = j),1) : dic_positions_associated[j]/6 for j in dic_positions_associated.keys()}
            d[state] = Categorical(sr_probs_map)
        d[StateSnakeAndLadder(position = self.grid_size)] = None
        return d


################################################################################
################## Classes and Functions for Question 5#########################
################################################################################

@dataclass(frozen=True)
class StateMP1:
    price: int


@dataclass
class StockPriceMP1(MarkovProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)

    def transition(self, state: StateMP1) -> Categorical[StateMP1]:
        up_p = self.up_prob(state)

        return Categorical({
            StateMP1(state.price + 1): up_p,
            StateMP1(state.price - 1): 1 - up_p
        })

@dataclass
class StockPriceMP1Reward(MarkovRewardProcess[StateMP1]):

    level_param: int  # level to which price mean-reverts
    alpha1: float = 0.25  # strength of mean-reversion (non-negative value)

    def up_prob(self, state: StateMP1) -> float:
        return get_logistic_func(self.alpha1)(self.level_param - state.price)
    
    def f(self, state: StateMP1) -> float:
        #We chose here as an example to define the reward as equal to the price
        #We can change f here to change the way we define the rewards
        return float(state.price)
    
    def transition_reward(
            self,
            state: StateMP1) -> SampledDistribution[Tuple[StateMP1,float]]:
        def sample_next_state_reward(state = state) -> Tuple[StateMP1,float]:
            up_p = self.up_prob(state)
            if np.random.random()<up_p:
                next_state: StateMP1 = StateMP1(state.price + 1)
            else:
                next_state: StateMP1 = StateMP1(state.price-1)
            reward: float = self.f(next_state)
            return next_state,reward
        return SampledDistribution(sample_next_state_reward)

def process1_reward_traces(
    start_price: int,
    level_param: int,
    alpha1: float,
    time_steps: int,
    num_traces: int
) -> np.ndarray:
    mp = StockPriceMP1Reward(level_param=level_param, alpha1=alpha1)
    start_state_distribution = Constant(StateMP1(price=start_price))
    return np.vstack([
        np.fromiter((s.reward for s in itertools.islice(
            mp.simulate_reward(start_state_distribution),
            time_steps + 1
        )), float) for _ in range(num_traces)])

def compute_value_function(
    start_price: int,
    level_param: int,
    alpha1: float,
    time_steps: int,
    num_traces: int,
    gamma: int
        ) -> float:
    rewards_traces: np.ndarray = process1_reward_traces(start_price = start_price,
                                            level_param = level_param,
                                            alpha1 = alpha1,
                                            time_steps = time_steps,
                                            num_traces = num_traces)
    value_iteration: list = []
    for i in rewards_traces:
        value = 0
        for j in range(len(i)):
            value += i[j]*gamma**j
        value_iteration+=[value]
    return np.mean(np.array(value_iteration))
        



if __name__ == '__main__':
    #QUESTIONS 1/2
    #Example of game (we took the example given in this Piazza post: https://piazza.com/class/kjgrmumjhtc3br?cid=17)
    grid: Iterable[SnakesAndLadders] = []
    list_start_pos = [3,7,12,20,25,28,31,38,45,49,53,60,65,67,69,70,76,77,82,88,94,98]
    list_end_pos = [39,48,51,41,57,35,6,1,74,8,17,85,14,90,92,34,37,83,63,50,42,54]
    for i in range(len(list_start_pos)):
        grid+=[SnakesAndLadders(start = list_start_pos[i],end = list_end_pos[i])]
    grid_size = 100
    #Below it corresponds to the execution of question 1/2

    game = SnakesAndLaddersGame(grid_size = grid_size,grid = grid)
    print("Transition Map")
    print("--------------")
    print(game)
    
    #We use the process_traces function to get access to a distribution of time steps to finish the game
    array_length = process_traces(10000,10000,game)
    x,y = get_terminal_histogram(array_length)

    plot_list_of_curves(
        [x],
        [y],
        ["r"],
        [
            r"Snakes and Ladders Game"
        ],
        "Time Steps to finish the game",
        "Counts",
        "Distribution of the time steps to finish the game"
    )

    #QUESTION 4
    game_reward = SnakesAndLaddersRewards(grid_size = grid_size, grid = grid)
    expected_steps = game_reward.get_value_function_vec(gamma = 1)
    print(expected_steps)
    
    #QUESTION 5
    rewards = process1_reward_traces(start_price = 100, level_param = 100, alpha1 = 0.25, time_steps = 100, num_traces = 100)
    value_function = compute_value_function(start_price = 100,
                                            level_param = 100,
                                            alpha1 = 0.25,
                                            time_steps = 100,
                                            num_traces = 1000,
                                            gamma = 0.9)

