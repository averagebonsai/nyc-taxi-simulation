Next, I want to run another experiment where the independent variable is swapping out the demand function. the code should be compatible to running experiments with either demand function. 

choose latent_demand to be equal to the latent poisson parameter from @data/latent_mle_params.csv corresponding to (PULocationID, day_of_week, request_hour) = (4, 4, 23). 

let baseline_price = avg(baseline_fare for all rides beginning from region 4) (data from graph.csv)

quantity demanded for agent i = latent_demand x e^-theta * (s_i - 1) / sum(e^-theta * (s_j - 1)) + e^(-theta * (3 / baseline_price) - 1)
- this follows a softmax distribution. 
- sum j is from 1 to n. 
- the last term with the baseline price is effectively the outside option: what if people want to take the subway. 

all other terms is as usual. profit = (p_i - c)q_i, where q_i is the quantity demanded, and that both p_i is directly proportional to the surge multiplier. 

clarify before implementing. 