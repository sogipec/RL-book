from dataclasses import dataclass
from typing import Callable, Tuple, Iterator, Sequence, List,Optional
import numpy as np
from rl.dynamic_programming import V
from scipy.stats import norm
from rl.markov_decision_process import FinitePolicy
from rl.distribution import Constant, Categorical
from rl.finite_horizon import optimal_vf_and_policy
from rl.distribution import Distribution, SampledDistribution, Choose, Gaussian,Uniform
from rl.markov_decision_process import MarkovDecisionProcess, Policy
from rl.function_approx import DNNSpec, AdamGradient, DNNApprox,LinearFunctionApprox,FunctionApprox
from rl.approximate_dynamic_programming import back_opt_vf_and_policy,value_iteration
from rl.approximate_dynamic_programming import back_opt_qvf
from operator import itemgetter
from rl.chapter9.order_book import OrderBook, DollarsAndShares,PriceSizePairs
from rl.markov_process import MarkovProcess
from numpy.random import poisson

#PROBLEM 1
@dataclass(frozen=True)
class OrderBookMPModel1(MarkovProcess[OrderBook]):
    #In the first model, the incoming order is either a market or a limit order with proba prop_market_order
    #It is a buy order with proba prop_buy_order
    #The number of shares bought is the mean of all the orders (bid and asks)
    #And the price of the limit orders is chosen randomly among one of the prices of the bid or ask amounts
    prob_buy_order: float
    prob_market_order: float    
    
    def transition(self,state:OrderBook) -> Optional[Distribution[OrderBook]]:
        descending_bids: PriceSizePairs = state.descending_bids
        ascending_asks: PriceSizePairs = state.ascending_asks
        volume:int = 0
        count_orders:float = 0
        list_bid_amounts: Sequence[float] = []
        list_ask_amounts: Sequence[float] = []
        for i in descending_bids:
            volume += i.shares
            count_orders += 1
            list_bid_amounts.append(i.dollars)
        for i in ascending_asks:
            volume += i.dollars
            count_orders += 1
            list_ask_amounts.append(i.dollars)
        num_shares_by_model:int = volume//count_orders
        
        if len(list_bid_amounts) == 0 or len(list_ask_amounts) == 0:
            return None
        
        def sr_sampler_func(
            list_bid_amounts = list_bid_amounts,
            list_ask_amounts = list_ask_amounts,
            num_shares_by_model = num_shares_by_model
        ) -> OrderBook:
            
            if np.random.random()<self.prob_buy_order:
                print("Buy")
                if np.random.random() < self.prob_market_order:
                    print("Market")
                    _,new_state = state.buy_market_order(num_shares_by_model)
                else:
                    print("Limit")
                    price = np.random.choice(list_bid_amounts)
                    new_state = state.buy_limit_order(price, num_shares_by_model)
            else:
                print("Sell")
                if np.random.random() < self.prob_market_order:
                    print("Market")
                    _,new_state = state.sell_market_order(num_shares_by_model)
                else:
                    print("Limit")
                    price = np.random.choice(list_ask_amounts)
                    new_state = state.sell_limit_order(price, num_shares_by_model)                
            return new_state

        return SampledDistribution(
            sampler=sr_sampler_func,
            expectation_samples=1000
        )

@dataclass(frozen=True)
class OrderBookMPModel2(MarkovProcess[OrderBook]):
    #In this second model, the incoming order is either a market or a limit order with proba prop_market_order
    #It is a buy order with proba prop_buy_order
    #The number of shares bought is chosen randomly in a range which limits are a function of the max shares offered and min shares offered in the market
    #And the price of the limit orders is chosen randomly among one of the prices of the bid or ask amounts with a proba 1-prob_closer
    #With a proba prob_closer, the price of the limit orders is chosen randomly in the spread
    prob_buy_order: float
    prob_market_order: float
    prob_closer:float    
    
    def transition(self,state:OrderBook) -> Optional[Distribution[OrderBook]]:
        descending_bids: PriceSizePairs = state.descending_bids
        ascending_asks: PriceSizePairs = state.ascending_asks
        max_shares = 0
        min_shares = np.inf
        list_bid_amounts: Sequence[float] = []
        list_ask_amounts: Sequence[float] = []
        for i in descending_bids:
            if i.shares>max_shares:
                max_shares = i.shares
            if i.shares<min_shares:
                min_shares = i.shares
            list_bid_amounts.append(i.dollars)
        for i in ascending_asks:
            if i.shares>max_shares:
                max_shares = i.shares
            if i.shares<min_shares:
                min_shares = i.shares
            list_ask_amounts.append(i.dollars)
        if len(list_bid_amounts) == 0 or len(list_ask_amounts) == 0:
            return None
        
        def sr_sampler_func(
            list_bid_amounts = list_bid_amounts,
            list_ask_amounts = list_ask_amounts,
            max_shares = max_shares,
            min_shares = min_shares
        ) -> OrderBook:
            num_shares_by_model = np.random.randint(min(min_shares//2-10,1),max_shares*2+10)
            list_closer_amounts:list = []
            for i in range(0,11):
                list_closer_amounts += [state.bid_price()+ state.bid_ask_spread()*i/10]
            
            if np.random.random()<self.prob_buy_order:
                print("Buy")
                if np.random.random() < self.prob_market_order:
                    print("Market")
                    _,new_state = state.buy_market_order(num_shares_by_model)
                else:
                    print("Limit")
                    if np.random.random() >= self.prob_closer:
                        price = np.random.choice(list_bid_amounts)
                    else:
                        price = np.random.choice(list_closer_amounts)
                    new_state = state.buy_limit_order(price, num_shares_by_model)
            else:
                print("Sell")
                if np.random.random() < self.prob_market_order:
                    print("Market")
                    _,new_state = state.sell_market_order(num_shares_by_model)
                else:
                    print("Limit")
                    if np.random.random() >= self.prob_closer:
                        price = np.random.choice(list_ask_amounts)
                    else:
                        price = np.random.choice(list_closer_amounts)
                    new_state = state.sell_limit_order(price, num_shares_by_model)                
            return new_state

        return SampledDistribution(
            sampler=sr_sampler_func,
            expectation_samples=1000
        )
    
@dataclass(frozen=True)
class OrderBookMPModel3(MarkovProcess[OrderBook]):
    #In this third model, the incoming order is either a market or a limit order with proba prop_market_order
    #It is a buy order with proba prop_buy_order
    #The number of shares bought is chosen randomly in a range which limits are a function of the max shares offered and min shares offered in the market
    #And the price of the limit orders is chosen randomly among one of the prices of the bid or ask amounts with a proba 1-prob_closer which depends on the spread
    #With a proba prob_closer, the price of the limit orders is chosen randomly in the spread
    prob_buy_order: float
    prob_market_order: float  
    
    def transition(self,state:OrderBook) -> Optional[Distribution[OrderBook]]:
        descending_bids: PriceSizePairs = state.descending_bids
        ascending_asks: PriceSizePairs = state.ascending_asks
        max_shares = 0
        min_shares = np.inf
        list_bid_amounts: Sequence[float] = []
        list_ask_amounts: Sequence[float] = []
        for i in descending_bids:
            if i.shares>max_shares:
                max_shares = i.shares
            if i.shares<min_shares:
                min_shares = i.shares
            list_bid_amounts.append(i.dollars)
        for i in ascending_asks:
            if i.shares>max_shares:
                max_shares = i.shares
            if i.shares<min_shares:
                min_shares = i.shares
            list_ask_amounts.append(i.dollars)
        if len(list_bid_amounts) == 0 or len(list_ask_amounts) == 0:
            return None
        
        def sr_sampler_func(
            list_bid_amounts = list_bid_amounts,
            list_ask_amounts = list_ask_amounts,
            max_shares = max_shares,
            min_shares = min_shares
        ) -> OrderBook:
            num_shares_by_model = np.random.randint(min(min_shares,1),max_shares+30)
            if state.bid_ask_spread()>10:
                prob_closer = 1
            else:
                prob_closer = state.bid_ask_spread()/10
            list_closer_amounts:list = []
            for i in range(0,11):
                list_closer_amounts += [state.bid_price()+ state.bid_ask_spread()*i/10]
            
            if np.random.random()<self.prob_buy_order:
                print("Buy")
                if np.random.random() < self.prob_market_order:
                    print("Market")
                    _,new_state = state.buy_market_order(num_shares_by_model)
                else:
                    print("Limit")
                    if np.random.random() >= prob_closer:
                        price = np.random.choice(list_bid_amounts)
                    else:
                        price = np.random.choice(list_closer_amounts)
                    new_state = state.buy_limit_order(price, num_shares_by_model)
            else:
                print("Sell")
                if np.random.random() < self.prob_market_order:
                    print("Market")
                    _,new_state = state.sell_market_order(num_shares_by_model)
                else:
                    print("Limit")
                    if np.random.random() >= prob_closer:
                        price = np.random.choice(list_ask_amounts)
                    else:
                        price = np.random.choice(list_closer_amounts)
                    new_state = state.sell_limit_order(price, num_shares_by_model)                
            return new_state

        return SampledDistribution(
            sampler=sr_sampler_func,
            expectation_samples=1000
        )

@dataclass(frozen=True)
class PriceAndShares:
    price: float
    shares: int
    x:float

@dataclass(frozen=True)
class PriceAndSharesAndX:
    price: float
    shares: int
    x: float

#PROBLEM 2
@dataclass(frozen=True)
class OptimalOrderExecutionCustomized:
    '''
    shares refers to the total number of shares N to be sold over
    T time steps.

    time_steps refers to the number of time steps T.

    avg_exec_price_diff refers to the time-sequenced functions g_t
    that gives the average reduction in the price obtained by the
    Market Order at time t due to eating into the Buy LOs. g_t is
    a function of PriceAndShares that represents the pair of Price P_t
    and MO size N_t. Sales Proceeds = N_t*(P_t - g_t(P_t, N_t)).

    price_dynamics refers to the time-sequenced functions f_t that
    represents the price dynamics: P_{t+1} ~ f_t(P_t, N_t). f_t
    outputs a distribution of prices.

    utility_func refers to the Utility of Sales proceeds function,
    incorporating any risk-aversion.

    discount_factor refers to the discount factor gamma.

    func_approx refers to the FunctionApprox required to approximate
    the Value Function for each time step.

    initial_price_distribution refers to the distribution of prices
    at time 0 (needed to generate the samples of states at each time step,
    needed in the approximate backward induction algorithm).
    '''
    shares: int
    time_steps: int
    avg_exec_price_diff: Sequence[Callable[[PriceAndShares], float]]
    price_dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]]
    pho: float
    utility_func: Callable[[float], float]
    discount_factor: float
    func_approx: FunctionApprox[PriceAndShares]
    initial_price_distribution: Distribution[float]
    init_x_distrib: Distribution[float]

    def get_mdp(self, t: int) -> MarkovDecisionProcess[PriceAndShares, int]:
        """
        State is (Price P_t, Remaining Shares R_t)
        Action is shares sold N_t
        """

        utility_f: Callable[[float], float] = self.utility_func
        price_diff: Sequence[Callable[[PriceAndShares], float]] = \
            self.avg_exec_price_diff
        dynamics: Sequence[Callable[[PriceAndShares], Distribution[float]]] = \
            self.price_dynamics
        pho = self.pho
        steps: int = self.time_steps

        class OptimalExecutionMDP(MarkovDecisionProcess[PriceAndShares, int]):

            def step(
                self,
                p_r: PriceAndShares,
                sell: int
            ) -> SampledDistribution[Tuple[PriceAndShares, float]]:

                def sr_sampler_func(
                    p_r=p_r,
                    sell=sell
                ) -> Tuple[PriceAndShares, float]:
                    
                    p_s: PriceAndShares = PriceAndShares(
                        price=p_r.price,
                        shares=sell,
                        x = x
                    )
                    next_price: float = dynamics[t](p_s).sample()
                    next_rem: int = p_r.shares - sell
                    next_x = pho*p_r.x+Uniform().sample()
                    next_state: PriceAndShares = PriceAndShares(
                        price=next_price,
                        shares=next_rem,
                        x = next_x
                    )
                    reward: float = utility_f(
                        sell * (p_r.price - price_diff[t](p_s))
                    )
                    return (next_state, reward)

                return SampledDistribution(
                    sampler=sr_sampler_func,
                    expectation_samples=100
                )

            def actions(self, p_s: PriceAndShares) -> Iterator[int]:
                if t == steps - 1:
                    return iter([p_s.shares])
                else:
                    return iter(range(p_s.shares + 1))

        return OptimalExecutionMDP()

    def get_states_distribution(self, t: int) -> \
            SampledDistribution[PriceAndShares]:

        def states_sampler_func() -> PriceAndShares:
            price: float = self.initial_price_distribution.sample()
            rem: int = self.shares
            x:float = self.init_x_distrib.sample()
            for i in range(t):
                sell: int = Choose(set(range(rem + 1))).sample()
                price = self.price_dynamics[i](PriceAndShares(
                    price=price,
                    shares=rem,
                    x = x
                )).sample()
                rem -= sell
                new_x = self.pho*x+Uniform().sample()
            return PriceAndShares(
                price=price,
                shares=rem,
                x = new_x
            )

        return SampledDistribution(states_sampler_func)

    def backward_induction_vf_and_pi(
        self
    ) -> Iterator[Tuple[FunctionApprox[PriceAndShares],
                        Policy[PriceAndShares, int]]]:

        mdp_f0_mu_triples: Sequence[Tuple[
            MarkovDecisionProcess[PriceAndShares, int],
            FunctionApprox[PriceAndShares],
            SampledDistribution[PriceAndShares]
        ]] = [(
            self.get_mdp(i),
            self.func_approx,
            self.get_states_distribution(i)
        ) for i in range(self.time_steps)]

        num_state_samples: int = 10000
        error_tolerance: float = 1e-6

        return back_opt_vf_and_policy(
            mdp_f0_mu_triples=mdp_f0_mu_triples,
            γ=self.discount_factor,
            num_state_samples=num_state_samples,
            error_tolerance=error_tolerance
        )
                    
if __name__ == '__main__':
    """
    print("Solving Problem 1")
    print("Model 1")
    model1 = OrderBookMPModel1(prob_buy_order = 0.5, prob_market_order = 0.5)
    bids: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (100 - x) * 10)
    ) for x in range(100, 90, -1)]
    asks: PriceSizePairs = [DollarsAndShares(
        dollars=x,
        shares=poisson(100. - (x - 105) * 10)
    ) for x in range(105, 115, 1)]

    ob0: OrderBook = OrderBook(descending_bids=bids, ascending_asks=asks)
    ob0.pretty_print_order_book()
    ob0.display_order_book()
    start_state_distribution = Constant(ob0)

    simulation = model1.simulate(start_state_distribution)
    count = 0
    for i in simulation:
        count+=1
        i.display_order_book()
        i.pretty_print_order_book()
        if count>30:
            break

    print("Model 2")
    model2 = OrderBookMPModel2(prob_buy_order = 0.5, prob_market_order = 0.5,prob_closer = 0.8)
    simulation = model2.simulate(start_state_distribution)
    count = 0
    for i in simulation:
        count+=1
        i.display_order_book()
        i.pretty_print_order_book()
        if count>30:
            break 

    print("Model 3")
    model3 = OrderBookMPModel3(prob_buy_order = 0.5, prob_market_order = 0.5)
    simulation = model3.simulate(start_state_distribution)
    count = 0
    for i in simulation:
        count+=1
        i.display_order_book()
        i.pretty_print_order_book()
        if count>30:
            break 
    """
    print("Solving Problem 2")

    init_price_mean: float = 100.0
    init_price_stdev: float = 10.0
    num_shares: int = 100
    x:int = 1
    num_time_steps: int = 5
    alpha: float = 0.03
    beta: float = 0.05
    mu_z:float = 0.
    sigma_z: float = 1.
    theta: float = 0.05
    pho: float = 1.

    price_diff = [lambda p_s: beta * p_s.shares * p_s.price - theta*p_s.price*p_s.x for _ in range(num_time_steps)]
    dynamics = [lambda p_s: Gaussian(
        μ=p_s.price*mu_z,
        σ=p_s.price**2*sigma_z
    ) for _ in range(num_time_steps)]
    #dynamics_x = [lambda p_s: p_s.x*pho + Uniform() for _ in range(num_time_steps)]
    ffs = [
        lambda p_s: p_s.price * p_s.shares,
        lambda p_s: float(p_s.shares * p_s.shares)
    ]
    fa: FunctionApprox = LinearFunctionApprox.create(feature_functions=ffs)
    init_price_distrib: Gaussian = Gaussian(
        μ=init_price_mean,
        σ=init_price_stdev
    )
    init_x_distrib: Gaussian = Constant(x)

    ooe: OptimalOrderExecutionCustomized = OptimalOrderExecutionCustomized(
        shares=num_shares,
        time_steps=num_time_steps,
        avg_exec_price_diff=price_diff,
        price_dynamics=dynamics,
        pho = pho,
        utility_func=lambda x: x,
        discount_factor=1,
        func_approx=fa,
        initial_price_distribution=init_price_distrib,
        init_x_distrib=init_x_distrib
    )
    it_vf: Iterator[Tuple[FunctionApprox[PriceAndShares],
                          Policy[PriceAndShares, int]]] = \
        ooe.backward_induction_vf_and_pi()

    state: PriceAndShares = PriceAndShares(
        price=init_price_mean,
        shares=num_shares,
        x = x
    )
    print("Backward Induction: VF And Policy")
    print("---------------------------------")
    print()
    for t, (v, p) in enumerate(it_vf):
        print(f"Time {t:d}")
        print()
        opt_sale: int = p.act(state).value
        val: float = v.evaluate([state])[0]
        print(f"Optimal Sales = {opt_sale:d}, Opt Val = {val:.3f}")
        print()
        print("Optimal Weights below:")
        print(v.weights.weights)
        print()
        