# Network Size Test

## Hypothesis
The size of the network affects the kind of function it can approximate. We don't know how complicated our value function is, so it is a good idea to:
1. try out various sizes for a single problem,
2. check if the optimal size varies for problems with different levels of complexity.

## Test
1. run cartpole with various network sizes
2. look at model performance across various problems

## Result
![aggregated](./aggregated_cheetah_run.png)
![aggregated](./aggregated_walker_run.png)
At the same order of magnitude the differences appear to be insignificant
