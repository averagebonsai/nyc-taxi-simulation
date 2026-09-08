# Direction for NYC Taxi Pricing Economic Experiments

### Research Question
Within a NYC Taxi ride-hailing duopoly, holding market conditions, action space and memory constant, how does choice of pricing-algorithm affect pricing and punishment behaviour in the simulation? 

### Motivating Paper 
Calvano et al (2020) describes the possibility of algorithmic collusion, between 2 (or more) traditional Q-learning agents. 

### Value-add 
Calvano et al (2020)'s paper has a deliberately narrow scope, to rigorously prove the possibility and existence of algorithmic collusion. Its setup is also relatively abstract, and not grounded in a real-world scenario. This exploratory paper seeks to build on Calvano et al's contributions in 2 main ways: 
(1) Reification: Do simplified versions of Calvano's experiments hold up when contextualised to a NYC Taxi duopoly / oligopoly? 
(2) Exploring a new independent variable: How does algorithm choice modify the nature of collusion in this set-up? 

### Immediate Task: Adapting Calvano's set-up 
The code folders for the Calvano set-up have been placed inside @src_calvano. It details the initial environment set-up, implementation of the q-learning algorithm, and an entry point in @src_calvano/main.py. 

The algorithms/init need to be contextualised to the NYC Taxi Code that's already been developed. Some bits can be removed, such as the demand-supply models (trained on actual data). Others may need to be implemented, such as the numerical computation of the monopoly price (if applicable). 

The goal for this step would be to run the q-learning algorithms written in @src_calvano/qlearning.py in the NYC Taxi simulation -- but we will truncate the game earlier (number of rounds would maximally be in the high hundreds range, especially since each episode is 120 steps).

The initial duopoly that Calvano sets is: 
- cost^i (marginal cost)= 1
- a^i - cost^i = 1 (a^i is a product quality, within the softmax model, higher a^i means higher quality, if all a^i the same, no product differentiation)
- a0 = 0 (outside option is not considered)
- mu = 1/4 (a measure of product stickiness, if one product is slightly better than the other, how much does demand change. if mu = 0, perfect substitutes, everyone will flock to the better/cheaper product)
- delta = 0.95 (discount factor per time horizon)
- m = 15 (number of price options)
- zye = 0.1 (how big the bins are between each price multiple)
- in the paper, k = 1 (one-period memory) <--> in the code, k refers to number of players. 

A normalised profit margin is calculated by doing a min-max normalisation between the nash-equilibrium price (p^n) and the monopoly price (p^m). 

We should expect the profit margin to be around 70-90% for a supermajority of set-ups. 

### Next Steps 


### Future Work (beyond the scope of this project)
It would be especially interesting to explore the behaviour of AI Pricing agents, especially since their performance is variable given their inherent non-determinism, as well as the kind of harness that surrounds the agents. 