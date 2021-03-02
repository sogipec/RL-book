from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List,Optional,Iterable,TypeVar,Mapping,Dict
import numpy as np
from rl.distribution import Constant, Categorical
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian,Uniform,Constant
from rl.markov_decision_process import MarkovDecisionProcess, Policy
import rl.markov_process as mp
from rl.returns import returns
from rl.chapter8.optimal_exercise_bin_tree import OptimalExerciseBinTree
from rl.markov_decision_process import FiniteMarkovDecisionProcess,policy_from_q
from rl.markov_decision_process import FinitePolicy, StateActionMapping,TransitionStep
from numpy.polynomial.laguerre import lagval

S = TypeVar('S')
A = TypeVar('A')

#PROBLEM 1

#Class to Solve Problem 1
class LSTDPb1:
    def __init__(
            self,
            features:Sequence[Callable[[S],float]],
            gamma: float,
            num_iter:int):
        self.gamma:float = gamma
        self.features:Sequence[Callable[[S],float]] = features
        self.A: np.ndarray = np.zeros((len(features),len(features)))
        self.b: np.ndarray = np.zeros(len(features))
        self.num_iter:int = num_iter
    
    def accumulate(self,data_point: TransitionStep[S,A]):
        state:S = data_point.state
        next_state:S = data_point.next_state
        reward:float = data_point.reward
        features_state: np.ndarray = np.array([phi(state) for phi in self.features])
        features_next_state: np.ndarray = np.array([phi(next_state) for phi in self.features])
        self.b += reward*features_state
        self.A += np.outer(features_state,features_state-self.gamma*features_next_state)
    
    def update(self,data:Iterable[TransitionStep[S,A]]) -> np.ndarray :
        count = 0
        for data_point in data:
            count += 1
            if count>self.num_iter:
                break
            self.accumulate(data_point)
        return np.dot(np.linalg.pinv(self.A),self.b)
    """
    def get_weights(self)->np.ndarray:
        return np.dot(np.linalg.pinv(self.A),self.b)
    """

#PROBLEM 2
#We start by defining some utility functions
class LSTDQ:
    def __init__(
            self,
            features:Sequence[Callable[[S,A],float]],
            gamma: float,
            num_iter:int,
            #batch_size:int,
            actions: Mapping[S, Sequence[A]],
            weights:np.ndarray):
        self.gamma:float = gamma
        self.features:Sequence[Callable[[S],float]] = features
        self.A: np.ndarray = np.zeros((len(features),len(features)))
        self.b: np.ndarray = np.zeros(len(features))
        self.num_iter:int = num_iter
        self.weights:np.ndarray = weights
        self.actions: Mapping[S, Sequence[A]] = actions
        #self.batch_size:int = batch_size
    
    def get_greedy_action(self,state:S)->Optional[A]:
        q_function = []
        if state not in self.actions:
            return None
        for a in self.actions.get(state):
            f = [phi(state, a) for phi in self.features]
            q_function.append(np.dot(f, self.weights))
        opt_index: int = np.argmax(q_function)[0]
        return self.actions[state][opt_index]
    
    def accumulate(self,data_point: TransitionStep[S,A]):
        state:S = data_point.state
        next_state:S = data_point.next_state
        reward:float = data_point.reward
        action:A = data_point.action
        next_action_greedy:A = self.get_greedy_action(next_state)
        features_state: np.ndarray = np.array([phi(state,action) for phi in self.features])
        features_next_state: np.ndarray = np.array([phi(next_state,next_action_greedy) for phi in self.features])
        self.b += reward*features_state
        self.A += np.outer(features_state,features_state-self.gamma*features_next_state)
    
    def update(self,data:Sequence[TransitionStep[S,A]]) -> np.ndarray :
        count = 0
        for i in range(self.num_iter):
            index = np.random.randint(len(data))
            #We sample from our data points-> in LSTDQ you do experience replay
            data_point = data[index]
            self.accumulate(data_point)
            """
            count += 1
            #Every batch_size Updates, we change the weights
            if count>=self.batch_size:
                self.weights = self.get_weights()
                count = 0
            """
        return np.dot(np.linalg.pinv(self.A),self.b)
    
    def get_weights(self)->np.ndarray:
        return np.dot(np.linalg.pinv(self.A),self.b)

#Function we use in Problem 2 to solve the whole problem
def LSPIPb2(
        #We choose here a sequence of data as a parameter rather than an Iterator
        #To be able to sample from it
        data: Sequence[TransitionStep[S,A]],
        features: Sequence[Callable[[S, A], float]],
        actions: Mapping[S, Sequence[A]],
        weights_0: np.ndarray,
        num_iter:int,
        #batch_size:int,
        gamma: float,
        tolerance: float
) -> Iterator[Callable[[S], A]]:
    m: int = len(features)
    epsilon: float = tolerance * 1e6
    weights = weights_0
    while epsilon >= tolerance:
        old_weights = weights
        #We alternate between LSTDQ and Policy Improvement
        lstdq = LSTDQ(features,gamma,num_iter,actions,weights)
        weights = lstdq.update(data)
        #weights = lstdq.get_weights()
        epsilon = max(abs(old_weights[i]-weights[i]) for i in range(m))
        yield lambda s: lstdq.get_greedy_action(s)
    
#PROBLEM 3
@dataclass(frozen=True)
class State:
    price:float
    time:float
    
@dataclass(frozen=True)
class Action:
    exercise:bool
    
def price_lspi(
    expiry_val: float,
    payoff: Callable[[float, float], float],
    gamma: float,
    data: Sequence[TransitionStep[State,Action]],
    weights: np.ndarray,
    features:Sequence[Callable[[State], float]]
) -> float:
    """
    Utility function we need to get the price of the option from the weights of the LSPI
    """
    prices = np.zeros(len(data))
    for i in range(len(data)):
        step = data[i]
        state = step.state
        price = state.price
        time = state.time
        exercise_price = payoff(time,price)
        if time == expiry_val:
            continue_price = 0.
        else:
            continue_price = weights.dot([phi(state) for phi in features])
        if exercise_price > continue_price:
            prices[i] = gamma**(-time)*exercise_price
            break
    return np.mean(prices)

#We will not use the interface we defined earlier (we could have redefined a class)
#But we prefered to work this way since the way we choose the next action and update
#A and b is not the same at each ste^p
def LSPIPb3(data: Sequence[TransitionStep[State,Action]],
            features: Sequence[Callable[[State], float]],
            #actions: Mapping[State, Sequence[Action]],
            weights_0: np.ndarray,
            num_iter:int,
            batch_size:int,
            gamma: float,
            #tolerance: float,
            expiry_val:float,
            payoff: Callable[[float, float], float]):
    
    weights = weights_0
    num_features:int = len(features)
    for _ in range(num_iter):
        A = np.zeros((num_features, num_features))
        b = np.zeros(num_features)
        for _ in range(batch_size):
            index = np.random.randint(len(data))
            #We sample from our data points-> in LSTDQ you do experience replay
            data_point = data[index]
            state:State = data_point.state
            next_state:State = data_point.next_state
            reward:float = 0.
            #action:Action = data_point.action
            features_state: np.ndarray = np.array([phi(state) for phi in features])
            features_next_state: np.ndarray = np.zeros(num_features)
            g_s = payoff(expiry_val,next_state.price)
            if next_state.time == expiry_val:
                reward = g_s
            else:
                potential_next_state = np.array([phi(next_state) for phi in features])
                if g_s>=weights.dot(features_next_state):
                    reward = g_s
                else:
                    features_next_state = potential_next_state
            A += np.outer(features_state,features_state-gamma*features_next_state)
            b += reward*gamma*features_next_state
        weights = np.dot(np.linalg.pinv(A),b)
    return price_lspi(expiry_val,payoff,gamma,data,features,weights)
    

if __name__ == '__main__':
    from rl.gen_utils.plot_funcs import plot_list_of_curves

    spot_price_val: float = 100.0
    strike: float = 100.0
    is_call: bool = False
    expiry_val: float = 1.0
    rate_val: float = 0.05
    vol_val: float = 0.25
    num_steps_val: int = 300

    if is_call:
        opt_payoff = lambda _, x: max(x - strike, 0)
    else:
        opt_payoff = lambda _, x: max(strike - x, 0)
    #We use Laguerre Polynomials
    laguerre_degree:int = 3
    eye = np.eye(laguerre_degree)
    def laguerre_feature_function(state: State, num_feature: int) -> float:
        xp = state.price / strike
        return np.exp(-xp / 2) * lagval(xp, eye[num_feature])
    
    #We have some features for the price and some features for the time
    def get_feature_func(state:State, num_feature: int) -> float:
        dt = expiry_val / num_steps_val
        t = state.time * dt
        if num_feature == 0:
            fun = 1.
        elif num_feature < laguerre_degree + 1:
            fun = laguerre_feature_function(state, num_feature - 1)
        elif num_feature == laguerre_degree + 1:
            fun = np.sin(-t * np.pi / (2. * expiry_val) + np.pi / 2.)
        elif num_feature == laguerre_degree + 2:
            fun = np.log(expiry_val - t)
        else:
            fun = (t / expiry_val)*(t / expiry_val)
        return fun
    
    features = [lambda state, i=i: get_feature_func(state, i) for i in
                     range(laguerre_degree + 4)]
    
    weights_0 = np.zeros(len(features))
    num_iter = 100
    batch_size = 10000
    gamma = 0.9
    #TODO
    #To test our implementation, we need to find a way to generate 
    data:Sequence[TransitionStep[State,Action]] = []
    #When defining our data, we need to be particularly wary of defining well 
    #our transition steps each time, and states with the correct time property
    #We don't care about the action we set in the transition step.
    
    price:float = LSPIPb3(data,
                            features,
                            #actions: Mapping[State, Sequence[Action]],
                            weights_0,
                            num_iter,
                            batch_size,
                            gamma,
                            #tolerance: float,
                            expiry_val,
                            opt_payoff)
    
    