# Direction for NYC Taxi Pricing Economic Experiments

## Research Question
Does tabular Q-learning sustain supra-competitive pricing when repeated duopoly competition is calibrated to a simplifed NYC Ride Hailing market? 

## Motivating Paper 
Calvano et al (2020) describes the possibility of algorithmic collusion, between 2 (or more) traditional Q-learning agents. Decentralised Q-learners can learn supra-competitive, punishment-supported strategies in their repeated-pricing environment.

## Value-add 
Calvano et al (2020)'s paper has a deliberately narrow scope, to rigorously prove the possibility and existence of algorithmic collusion. Its setup is also relatively abstract, and not grounded in a real-world scenario. This exploratory paper seeks to build on Calvano et al's contributions through reifying the simulation -- do these experiments hold up when contextualised to the NYC Taxi duopoly / oligopoly?  

## Immediate Task: Adapting Calvano's set-up 
The code folders for the Calvano set-up have been placed inside @src_calvano. It details the initial environment set-up, implementation of the q-learning algorithm, and an entry point in @src_calvano/main.py. 

The algorithms/init need to be contextualised to the NYC Taxi Code that's already been developed. Some bits can be removed, such as the demand-supply models (trained on actual data). Others may need to be implemented, such as the numerical computation of the monopoly price (if applicable). 

The goal for this step would be to run the q-learning algorithms written in @src_calvano/qlearning.py in the NYC Taxi simulation.

### Calvano's set-up
Each game involves only 2 firms, with demand following a softmax distribution (along with an "outside option"). Their first order conditions are calculated, and the monopoly and competitive (Bertrand Nash equilibrium) price calculated. A evenly-spaced range of prices (discrete grid) below the competitive price and above the monopoly price is created as the action space for the RL agent. 

The initial duopoly that Calvano sets is: 
- cost^i (marginal cost)= 1
- a^i - cost^i = 1 (a^i is a product quality, within the softmax model, higher a^i means higher quality, if all a^i the same, no product differentiation)
- a0 = 0 (outside option utility is normalised to zero)
- mu = 1/4 (a measure of product stickiness, if one product is slightly better than the other, how much does demand change. if mu = 0, perfect substitutes, everyone will flock to the better/cheaper product)
- delta = 0.95 (discount factor per time horizon)
- m = 15 (number of price options)
- xi = 0.1 (how big the bins are between each price multiple)
- in the paper, k = 1 (one-period memory) <--> in the code, k refers to number of discrete price actions. 

A normalised profit margin is calculated by doing a min-max normalisation between the nash-equilibrium price (p^n) and the monopoly price (p^m). 

We should expect the profit margin to be around 70-90% for a supermajority of set-ups. 

### Adapting the set-up to NYC Taxi Data 
First, demand and supply is not set arbitrarily, like in Calvano's model. Demand is calculated via MLE on the NYC Taxi Data. Supply of taxis is determined by a forward looking demand model -- but this needs to be re-calibrated. Importantly, demand but not supply is included in Calvano's model -- perhaps, this could be tailored, such that in the NYC Taxi Environment, Calvano's demand would refer to the gap between demand and supply for a particular zone. 

Second, the NYC Taxi Environment needs to introduce some marginal costs in order for both simulations to have "maximise profit" as their objective function. 

Third, unlike Calvano's set-up, the baseline monopoly and perfect competition prices cannot be directly derived via calculations. A Monte Carlo simulation is used to approximate the monopoly and perfect competition (mutual best response) prices. 

Finally, Calvano's model is limited to two agents in one environment. While the derivation of initial monopoly/competition benchmarks will be limited to a single taxi zone, the model will need to expand to managing the 262 taxi zones. To compensate for this, the number of runs per simulation would have to be greatly reduced. 

## Next Steps 


## Future Work (beyond the scope of this project)
It would be especially interesting to explore the behaviour of different algorithms, inclusive of LLM Pricing agents, especially since their performance is variable given their inherent non-determinism, as well as the kind of harness that surrounds the agents. 