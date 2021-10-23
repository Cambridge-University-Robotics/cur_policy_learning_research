# Network Depth Test

## Hypothesis
The depth of the network affects the kind of function it can approximate. We don't know how complicated our value function is, so it is a good idea to:
1. try out various depths for a single problem, 
2. check if the optimal depth varies for problems with different levels of complexity.

## Test
1. run cartpole with various network depths
2. look at model performance across various problems

## Result
![aggregated](./aggregated.png)

It is indeed the case that higher values are necessary to train the model properly. More analysis needs to be done regarding high gammas and instability. It might be the case that due to high value states not being reached often, that for complicated environments we don't need to care much about exploding values.