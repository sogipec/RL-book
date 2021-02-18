from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List,Optional,Iterable
import numpy as np
from rl.distribution import Constant, Categorical
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian,Uniform,Constant
from rl.markov_decision_process import MarkovDecisionProcess, Policy



@dataclass(frozen=True)
class State:
    #This is our state class that we call State.
    #A state is represented by a tuple (t,S,W,I)
    t:float
    S:float
    W:float
    I:float
    

@dataclass(frozen=True)    
class Action:
    #An action is defined by a Tuple(Pb,Pa)
    Pb:float
    Pa:float


class OptimalPolicy(Policy[State,Action]):
    k:float
    gamma:float
    sigma:float
    T:float
    
    def __init__(self,k,gamma,sigma,T):
        self.k = k
        self.gamma = gamma
        self.sigma = sigma
        self.T = T
        
    def act(self, state: State) -> Optional[Distribution[Action]]:
        delta_b:float = (2*state.I+1)*self.gamma*self.sigma**2*\
            (self.T-state.t)/2+1/self.gamma*np.log(1+self.gamma/self.k)
        delta_a:float = (-2*state.I+1)*self.gamma*self.sigma**2*\
            (self.T-state.t)/2+1/self.gamma*np.log(1+self.gamma/self.k)
        Pb:float = state.S-delta_b
        Pa:float = state.S+delta_a
        return Constant(Action(Pb = Pb,Pa = Pa))

def utility_func(x,gamma)->float:
    return -np.exp(-gamma*x)

class NaivePolicy(Policy[State,Action]):
    k:float
    gamma:float
    sigma:float
    T:float
    spread:float
    
    def __init__(self,k,gamma,sigma,T,spread):
        self.k = k
        self.gamma = gamma
        self.sigma = sigma
        self.T = T
        self.spread = spread
    def act(self, state: State) -> Optional[Distribution[Action]]:
        Pb:float = state.S-self.spread/2
        Pa:float = state.S+self.spread/2
        return Constant(Action(Pb = Pb,Pa = Pa))

    
#We define the MDP, the thing that will change is the 
class Problem2Optimal(MarkovDecisionProcess[State,Action]):
    k:float
    gamma:float
    sigma:float
    T:float
    delta_t:float
    c:float
    naive_spread:float
    def __init__(self,k,gamma,sigma,T,delta_t,c,naive_spread = 0):
        self.k:float = k
        self.gamma:float = gamma
        self.sigma:float = sigma
        self.T:float = T
        self.delta_t:float = delta_t
        self.c:float = c
        self.naive_spread = naive_spread
        
    def actions(self, state: State) -> Iterable[Action]:
        list_actions:Iterable[Action] = []
        #We define below the optimal policy
        delta_b:float = (2*state.I+1)*self.gamma*self.sigma**2*\
            (self.T-state.t)/2+1/self.gamma*np.log(1+self.gamma/self.k)
        delta_a:float = (-2*state.I+1)*self.gamma*self.sigma**2*\
            (self.T-state.t)/2+1/self.gamma*np.log(1+self.gamma/self.k)
        Pb:float = state.S-delta_b
        Pa:float = state.S+delta_a
        list_actions.append(Action(Pb = Pb,Pa = Pa))
        #We define the naive policy below
        Pb:float = state.S-self.naive_spread/2
        Pa:float = state.S+self.naive_spread/2
        list_actions.append(Action(Pb = Pb,Pa = Pa))
        return list_actions
    
    def step(
        self,
        state: State,
        action: Action
    ) -> Optional[Distribution[Tuple[State, float]]]:
        if state.t>self.T:
            return None    
        def sampler_func()->Tuple[State,float]:

            inventory = state.I
            PnL = state.W
            proba_inventory_d:float = self.c*np.exp(-self.k*(action.Pa-state.S))*self.delta_t
            if inventory>=1 and np.random.random()<proba_inventory_d:
                inventory -=1
                PnL+= action.Pa
            #print(action.Pb)
            proba_inventory_u:float = self.c*np.exp(-self.k*(-action.Pb+state.S))*self.delta_t   
            if np.random.random()<proba_inventory_u:
                inventory+=1
                PnL -= action.Pb
            ob_mid_price: float = state.S
            if np.random.random()<0.5:
                ob_mid_price += self.sigma*np.sqrt(self.delta_t)
            if np.random.random()<0.5:
                ob_mid_price -= self.sigma*np.sqrt(self.delta_t)
            next_state = State(t = state.t+self.delta_t,
                               S = ob_mid_price,
                               W = PnL,
                               I = inventory)
            if next_state.t >= self.T:
                reward = utility_func(next_state.W+next_state.I*next_state.S,self.gamma)
            else:
                reward = 0
            
            return (next_state,reward)
        
        return SampledDistribution(
            sampler=sampler_func,
            expectation_samples=1000
        )

def process_trace(trace,num_traces):
    """
    We're getting for each simulation in the traces the last reward, inventory, bid and ask
    """
    reward_arr = []
    inventory_arr = []
    bid_arr = []
    ask_arr = []
    count = 0
    for simulation in trace:
        count += 1
        if count%100 == 0:
            print(f"{count} traces processed")
        if count>num_traces:
            break
        for i in simulation:
            reward = i.reward
            inventory = i.state.I
            bid = i.action.Pb
            ask = i.action.Pa
        reward_arr.append(reward)
        inventory_arr.append(inventory)
        bid_arr.append(bid)
        ask_arr.append(ask)
    return np.array(reward_arr),np.array(inventory_arr),np.array(bid_arr),np.array(ask_arr)
    
    

if __name__ == '__main__':
    S = 100
    T = 1
    delta_t = 0.005
    gamma = 0.1
    I = 0
    k = 1.5
    c = 140
    sigma = 2
    optimal = Problem2Optimal(k = k,
                              gamma = gamma,
                              sigma = sigma,
                              T = T,
                              delta_t = delta_t,
                              c = c)
    init_state = State(t = 0,
                       S = S,
                       W = 0,
                       I = I)
    init_state_distrib = Constant(init_state)
    policy = OptimalPolicy(k = k,
                        gamma = gamma,
                        sigma = sigma,
                        T = T)
    """
    print("Computing the bid ask spread across different traces")
    traces = optimal.action_traces(init_state_distrib, policy)
    #simulation = optimal.simulate_actions(init_state_distrib, policy)
    count = 0
    
    bid_ask_array = []
    for simulation in traces:
        count += 1
        if count%100== 0:
            print(count)
        if count>1000:
            break
        else:
            for i in simulation:
                bid_ask = i.action.Pa-i.action.Pb
                bid_ask_array.append(bid_ask)
            #print(count2)
    bid_ask_array = np.array(bid_ask_array)
    bid_ask_spread_naive = np.mean(bid_ask_array)
    """
    bid_ask_spread_naive = 1.4917704227514237
    new_model = Problem2Optimal(k = k,
                              gamma = gamma,
                              sigma = sigma,
                              T = T,
                              delta_t = delta_t,
                              c = c,
                              naive_spread = bid_ask_spread_naive)
    naive_policy =  NaivePolicy(k = k,
                        gamma = gamma,
                        sigma = sigma,
                        T = T,
                        spread = bid_ask_spread_naive)
    traces_optimal = new_model.action_traces(init_state_distrib, policy)
    traces_naive = new_model.action_traces(init_state_distrib, naive_policy)
    print("Processing the traces for the Optimal Policy")
    reward_arr,inventory_arr,bid_arr,ask_arr = process_trace(traces_optimal, 1000)
    print("Processing the traces for the Naive Policy")
    reward_arr_n,inventory_arr_n,bid_arr_n,ask_arr_n = process_trace(traces_naive, 1000)
    print("Rewards")
    print("Optimal: ",np.mean(reward_arr))
    print("Naive: ",np.mean(reward_arr_n))
    print("Inventory")
    print("Optimal: ",np.mean(inventory_arr))
    print("Naive: ", np.mean(inventory_arr_n))
    print("Bid Price")
    print("Optimal: ",np.mean(bid_arr))
    print("Naive: ", np.mean(bid_arr_n))
    print("Ask Price")
    print("Optimal: ",np.mean(ask_arr))
    print("Naive: ", np.mean(ask_arr_n))
    
    
               