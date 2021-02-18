from dataclasses import dataclass
from typing import  List
from rl.distribution import Constant,Categorical
from rl.markov_decision_process import (FiniteMarkovDecisionProcess,
                                        StateActionMapping,
                                        ActionMapping)
from rl.dynamic_programming import (value_iteration_result)
from rl.markov_process import StateReward
from scipy.stats import poisson

@dataclass(frozen=True)
class State:
    #This is our state class that we call State.
    #A state is represented by a wage
    wage:int


@dataclass(frozen=True)    
class Action:
    #An action is defined by a Tuple(l,s)
    l: int
    s: int
    
class Problem3(FiniteMarkovDecisionProcess[State,Action]):
    def __init__(
        self,
        H:int,
        W:int,
        alpha: float,
        beta: float):
        
        self.H:int = H
        self.W:int = W
        self.alpha:float = alpha
        self.beta:float = beta
        
        super().__init__(self.get_mapping())
    def get_mapping(self) -> StateActionMapping[State, Action]:
        #We need to define the StateActionMapping for this Finite MDP
        mapping: StateActionMapping[State, Action] = {}
        list_actions:List[Action] = []
        #We start by defining all the available actions
        for i in range(self.H+1):
            range_j = self.H-i
            for j in range(range_j+1):
                list_actions.append(Action(i,j))
        self.list_actions:List[Action] = list_actions
        list_states:List[State] = []
        #Then we define all the possible states
        for i in range(1,self.W+1):
            list_states.append(State(i))
        self.list_states:List[State] = list_states
        for state in list_states:
            submapping:ActionMapping[Action,StateReward[State]] = {}
            for action in list_actions:
                s:int = action.s
                l:int = action.l
                reward:float = state.wage*(self.H-l-s)
                pois_mean:float = self.alpha*l
                proba_offer:float = self.beta*s/self.H
                if state.wage == self.W:
                    #If you're in state W, you stay in state W with constant
                    #Probability. The reward only depends on the action you
                    #you have chosen
                    submapping[action] = Constant((state,reward))
                elif state.wage == self.W-1:
                    #If you're in state W-1, you can either stay in your state
                    #or land in state W
                    submapping[action] = Categorical({
                        (state,
                         reward):
                            poisson.pmf(0,pois_mean)*(1-proba_offer),
                         (State(self.W),
                          reward):proba_offer+(1-proba_offer)*\
                             (1-poisson.pmf(0,pois_mean))
                        })
                else:
                    #If you're in any other state, you can land to any state
                    #Between your current state and W with probabilities
                    #as described before
                    dic_distrib = {}
                    dic_distrib[
                        (state,
                        reward)] = poisson.pmf(0,pois_mean)*(1-proba_offer)
                    dic_distrib[
                        (State(state.wage+1),
                         reward)] = proba_offer*poisson.cdf(1,pois_mean)\
                                +(1-proba_offer)*poisson.pmf(1,pois_mean)
                    for k in range(2,self.W-state.wage):
                        dic_distrib[
                        (State(state.wage+k),
                         reward)] = poisson.pmf(k,pois_mean)
                    dic_distrib[
                        (State(self.W),
                         reward)] = 1-poisson.cdf(self.W-state.wage-1,pois_mean)
                    submapping[action] = Categorical(dic_distrib)
            mapping[state] = submapping
        return mapping
                        
                        
if __name__ == '__main__':
    H = 10
    W = 30
    alpha = 0.08
    beta = 0.82
    gamma = 0.95
    print("Defining the model")
    model =  Problem3(H,W,alpha,beta)
    print("Value iteration algorithm")
    opt_val, opt_pol = value_iteration_result(model,gamma)
    print(opt_pol)       
            
        
"""
if state.wage == self.W:
    for action in list_actions:
        #If you're in state W, you stay in state W with constant
        #Probability. The reward only depends on the action you
        #you have chosen
        submapping[action] = Constant((State(state.wage),
                                       state.wage*\
                                       (self.H-action.l-action.s)))
elif state.wage == self.W-1:
    for action in list_actions:
        s:int = action.s
        l:int = action.l
        #If you're in state W-1, you can either stay in your state
        #or land in state W
        submapping[action] = Categorical({
            (State(state.wage),
             state.wage*(self.H-l-s)):
                np.exp(-self.alpha*l)*(1-self.beta*s/self.H),
             (State(state.wage+1),
              state.wage*(self.H-l-s)):self.beta*s/self.H+\
                 (1-self.beta*s/self.H)*(1-poisson.pmf(0,self.alpha*l))
            })
else:
    for action in list_actions:
        #If you're in any other state, you can land to any state
        #Between your current state and W with probabilities
        #as described before
        s:int = action.s
        l:int = action.l 
        reward:float = state.wage*(self.H-l-s)
        dic_distrib = {}
        dic_distrib[
            (State(state.wage),
            reward)] = np.exp(-self.alpha*l)*\
                                (1-self.beta*s/self.H)
        dic_distrib[
            (State(state.wage+1),
             reward)] = self.beta*s/self.H*poisson.cdf(1,self.alpha*l)\
                    +(1-self.beta*s/self.H)*poisson.pmf(1,self.alpha*l)
        for k in range(2,self.W-state.wage):
            dic_distrib[
            (State(state.wage+k),
             reward)] = poisson.pmf(k,self.alpha*l)
        dic_distrib[
            (State(self.W),
             reward)] = 1-poisson.cdf(self.W-state.wage,self.alpha*l)
        submapping[action] = Categorical(dic_distrib)
"""   
    