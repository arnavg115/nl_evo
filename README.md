# NL_EVO

This project is based on this [paper](https://www.pnas.org/doi/full/10.1073/pnas.1323208111). This project includes a way to represent these circuits as weight matrixes and is an implementation of many optimization algorithms to see which produces neural circuits closest to those seen in the neural circuits of crustaceans. All the algorithms are essentially tweaking weights in order to get to the optimal phi value of 0.25. The phi value is the phase difference between the "legs" of the crustaceans and the paper which this work is based on shows that 0.25 is the optimal phi value.

## Backpropagation

This is found in `./backprop` and implements backpropagation on the neural circuit. It uses the phi value for calculating the error and then uses `sympy` to calculate partial derivatives and to update the weights.

## Genetic Algorithm

This is found in `./genetic` and implements a simple genetic algorithm. Here the fitness is calculate using the phi value and it starts with 200 individuals. This implementation is a bit crude and requires updating.

## Reinforcement Learning/PPO

This is found in `./RL` and implements reinforcement learning using the stable baselines 3 library. I found the PPO model from openai works the best and I used. For this I implemented a custom environment and designed my own reward function.

## Comparison

I then used cosine distance to compare which algorithm is closest. In my preliminary testing it seems that Reinforcement learning produces circuits that are closest to those seen in nature. Sample testing code is present in `test.ipynb`
