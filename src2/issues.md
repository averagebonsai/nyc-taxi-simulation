1. fix invalid probabilities -- currently the demand model cannot take anything that is less than 1. but eventually, i'd like it to take smaller multipliers too. 
2. fix fleet allocation model -- currently the fleet allocation model is too "powerful" since taxi drivers have perfect knowledge of next location demand. 
3. restructure files into 3 folders: market (environment set-up), policies (RL algorithms) and experiments
4. define policy contract
- act(observation) -> actions, action_info
- observe(transition) -> None
- end_episode() -> None
- snapshot() -> FrozenPolicy
5. think about whether it should optimise revenue per taxi or profit per vehicle. doesn't include any drawbacks on trip-time, operating costs, etc. 
6. taxis reach their destination immediately. 
7. modify courthoud freezing, currently it freezes expected multipliers rather than frozen stochastic policies. 
--- e.g if policy is choose 1.0x 75% of time and 2.0x 25% of time, E(policy) = 1.25x, and the current freezing chooses 1.25x all the time. but we want it to choose 1.0x 75% of the time and 2.0x 25% of the time. 
8. missing explicit collusion outcome measure -- comparing competitive benchmark and monopoly benchmark. 
9. Multi-seed aggregation or confidence intervals missing. Run over different seeds. And report a range/confidence interval. 
10. Baseline training starts from same day/hour pattern. Also agents don't observe hour/day -- to be incorporated in algo. 
11. Fare matrix contains a negative fare. 
12. Do we need to include memory? Each agent having a certain amount of memory. e.g Q-learning: Only able to see the results of the last k episodes, instead of being able to view the entire Q-table? [Note: Underlying assumption is perfect memory, which is quite common -- Amazon allows checking of current and past prices via API, Uber has full oversight on its dynamic prices, etc.]
--- Memory is important, because the agents need to "know" that other agents deviated from the strategy. |S| = m^(nk), m = no. of actions, n = no of firms, k = number of periods. 


Variables that should be taken into account by the algo: 
- Demand 
- Supply 
- Location 
- Time 

Calvano et al baseline: 
- General tabular Q-learning
- Repeated price competition with bounded price-history memory. 


Hypotheses to test: 
- Grounded in a new environment -- Does Calvano's results change significantly? 
- Algorithm choice -- with bounded memory and fixed parameters, but different algorithms, does collusion change? 