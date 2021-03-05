from typing import Callable, Tuple, Iterator, Sequence, List,Optional,Iterable,TypeVar,Mapping,Dict
import numpy as np

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
    data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
    state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    mapping:ValueFunc = {}
    counts_per_state:Mapping[S,int] = {}
    for sample in state_return_samples:
        state:S = sample[0]
        return_:float = sample[1]
        if state in counts_per_state:
            counts_per_state[state]+=1
            n:int = counts_per_state[state]
            mapping[state] = 1/n*return_+mapping[state]*(n-1)/n
            #mapping[state]+= return_
        else:
            counts_per_state[state] = 1
            mapping[state] = return_
    #for state in mapping:
    #    mapping[state] = mapping[state]/counts_per_state[state]
    return mapping
        
        
def get_state_reward_next_state_samples(
    data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i+1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    proba_func:ProbFunc = {}
    reward_func:RewardFunc = {}
    counts_per_state:Mapping[S,int] = {}
    for sample in srs_samples:
        state:S = sample[0]
        reward:float = sample[1]
        next_state:S = sample[2]
        if state in counts_per_state:
            counts_per_state[state]+=1
            n:int = counts_per_state[state]
            reward_func[state] = 1/n*reward+reward_func[state]*(n-1)/n
            if next_state not in proba_func[state]:
                proba_func[state][next_state] = 1
            else:
                proba_func[state][next_state] += 1
        else:
            counts_per_state[state] = 1
            reward_func[state] = reward
            proba_func[state] = {next_state:1}
    for state in proba_func:
        n:int = counts_per_state[state]
        for next_state in proba_func[state]:
            proba_func[state][next_state] = proba_func[state][next_state]/n
    return proba_func,reward_func
        
        
def get_mrp_value_function(
    prob_func: ProbFunc,
    reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """
    states:Sequence[S] = [s for s in reward_func]
    R:np.ndarray = np.array([reward_func[s] for s in states])
    to_invert:np.ndarray = np.zeros((len(states),len(states)))
    for i in range(len(states)):
        for j in range(len(states)):
            #We assume gamma = 1
            to_invert[i,j] = (i==j)-prob_func[states[i]].get(states[j],0)
    value_array:np.ndarray = np.dot(np.linalg.inv(to_invert),R)
    value_func:ValueFunc = {}
    for i in range(len(states)):
        value_func[states[i]] = value_array[i]
    return value_func
    
    

def get_td_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    num_updates: int = 300000,
    learning_rate: float = 0.3,
    learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """
    mapping:ValueFunc = {}
    counts_per_state:Mapping[S,int] = {}
    for i in range(num_updates):
        sample = srs_samples[np.random.randint(len(srs_samples))]
        state:S = sample[0]
        reward:float = sample[1]
        next_state:S = sample[2]
        if state in counts_per_state:
            counts_per_state[state]+=1
            n:int = counts_per_state[state]
        else:
            counts_per_state[state] = 1
            mapping[state] = 0
            n:int = 1
        alpha = learning_rate * (n / learning_rate_decay + 1) ** -0.5
        mapping[state] += alpha*(reward+mapping.get(next_state,0)-mapping[state])
    return mapping
    


def get_lstd_value_function(
    srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    states:Sequence[S] = []
    for sample in srs_samples:
        state:S = sample[0]
        if state not in states:
            states.append(state)
    A:np.ndarray = np.zeros((len(states),len(states)))
    b:np.ndarray = np.zeros(len(states))
    for sample in srs_samples:
        state:S = sample[0]
        reward:float = sample[1]
        next_state:S = sample[2]
        for i in range(len(states)):
            if states[i]==state:
                index_state = i
            if states[i] == next_state:
                index_next_state = i
        phi_s = np.zeros(len(states))
        phi_nexts = np.zeros(len(states))
        if next_state in states:
            phi_nexts[index_next_state] = 1
        phi_s[index_state] = 1
        b += reward*phi_s
        A += np.outer(phi_s,phi_s-phi_nexts)
    #print(A)
    value_array:np.ndarray = np.dot(np.linalg.inv(A),b)
    """
    #Tentative Approach in which we tried to do it with dictionnaries
    b:Mapping[S, float] = {}
    A:Mapping[S, Mapping[S, int]] = {}
    for sample in srs_samples:
        state:S = sample[0]
        reward:float = sample[1]
        next_state:S = sample[2]
        if state not in b:
            b[state] = reward
        else:
            b[state] += reward
        if state not in A:
            A[state] = {state:1,next_state :-1}
        else:
            A[state][state] += 1
            if next_state not in A[state]:
                A[state][next_state] = -1
            else:
                A[state][next_state] -= 1
    states:Sequence[S] = [state for state in b]
    b_matrix:np.ndarray = np.array([b[state] for state in states])
    A_matrix = np.zeros((len(states),len(states)))
    for i in range(len(states)):
        for j in range(len(states)):
            A_matrix[i,j] = A[states[i]].get(states[j],0)
    print(A_matrix)
    value_array:np.ndarray = np.dot(np.linalg.inv(A_matrix),b_matrix)
    """
    value_func:ValueFunc = {}
    for i in range(len(states)):
        value_func[states[i]] = value_array[i]
    return value_func
    

def get_tdc_value_function(
    srs_samples: Sequence[Tuple[S, float, S]],
    features:Sequence[Callable[[S],float]],
    learning_rate_alpha: float = 0.3,
    learning_rate_beta: float = 0.3,
    learning_rate_decay: int = 30,
    gamma:float = 1,
    num_updates:int = 300000
) -> ValueFunc:
    w = np.zeros(len(features))
    theta = np.zeros(len(features))
    counts_per_state:Mapping[S,int] = {}
    for i in range(num_updates):
    #for sample in srs_samples:
        #If we don't do experience replay in this case, there are not enough samples
        #for w to converge
        sample = srs_samples[np.random.randint(len(srs_samples))]
        state:S = sample[0]
        reward:float = sample[1]
        next_state:S = sample[2]
        if state in counts_per_state:
            counts_per_state[state]+=1
        else:
            counts_per_state[state] = 1
        n:int = counts_per_state[state]
        int_value:float = (n / learning_rate_decay + 1) ** -0.5
        alpha = learning_rate_alpha * int_value
        beta = learning_rate_beta * int_value
        features_next_state: np.ndarray = np.array([phi(next_state) for phi in features])
        features_state: np.ndarray = np.array([phi(state) for phi in features])
        delta:float = reward + gamma*np.dot(features_next_state,w)- np.dot(features_state,w)
        w += alpha*delta*features_state - alpha*gamma* features_next_state*np.dot(theta,features_state)
        theta += beta*(delta - np.dot(theta,features_state))*features_state
        #print(w)
        #print(theta)
    value_func:ValueFunc = {}
    for state in counts_per_state:
        features_state: np.ndarray = np.array([phi(state) for phi in features])
        value_func[state] = np.dot(features_state,w)
    return value_func
    
    

    


if __name__ == '__main__':
    given_data: DataType = [
        [('A', 2.), ('A', 6.), ('B', 1.), ('B', 2.)],
        [('A', 3.), ('B', 2.), ('A', 4.), ('B', 2.), ('B', 0.)],
        [('B', 3.), ('B', 6.), ('A', 1.), ('B', 1.)],
        [('A', 0.), ('B', 2.), ('A', 4.), ('B', 4.), ('B', 2.), ('B', 3.)],
        [('B', 8.), ('B', 2.)]
    ]

    sr_samps = get_state_return_samples(given_data)

    print("------------- MONTE CARLO VALUE FUNCTION --------------")
    print(get_mc_value_function(sr_samps))

    srs_samps = get_state_reward_next_state_samples(given_data)

    pfunc, rfunc = get_probability_and_reward_functions(srs_samps)

    print("-------------- MRP VALUE FUNCTION ----------")
    print(get_mrp_value_function(pfunc, rfunc))

    print("------------- TD VALUE FUNCTION --------------")
    print(get_td_value_function(srs_samps))

    print("------------- LSTD VALUE FUNCTION --------------")
    print(get_lstd_value_function(srs_samps))

    #PROBLEM 2
    print("------------- TDC VALUE FUNCTION --------------")
    states = ['A','B']
    #features = [lambda state: 1*(state==states[i]) for i in range(len(states))]
    features = [lambda state: 1*(state=='A'),
                lambda state: 1*(state=='B')]
    print(get_tdc_value_function(srs_samps,features))
    