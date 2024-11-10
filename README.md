# Ball Allocation Experiment

This repository provides code to simulate the allocation of balls into bins under various strategies. The program computes the load gap, which measures the difference between the maximum load of bins and the average, over multiple repetitions.

## Prerequisites

To compile and run this program, you need:
'''
- Python 3.x
- Required Python packages:
  - `numpy`
  - `matplotlib`
'''

## Running Ball Allocation Simulation

To simulate the allocation of balls into bins and compute the load gap, use the `simulate_ball_allocation` function. Below is an example setup for running the simulation with `10` bins and `100` balls:

'''
m = 10        # Number of bins
n = 100       # Number of balls
T = 10        # Number of repetitions
strategy = "two_choice"  # Choose allocation strategy

# Run the simulation
average_gap, std_dev_gap = simulate_ball_allocation(m, n, T, strategy=strategy)
print("Average Gap:", average_gap)
print("Standard Deviation of Gap:", std_dev_gap)
'''
## Exploring Different Strategies and Parameters

The `run_allocation_experiments` function lets you run experiments on various allocation strategies, numbers of bins, and numbers of balls. You can specify multiple values for `m`, `n`, and other parameters to see how they impact the load gap.

Run the experiments with:

  results = run_allocation_experiments(m_values=[10, 100], n_factors=[1, 5, 10], T=10)

The program includes a plotting functions to visualize the results of the experiments. To plot the load gap as a function of the number of balls for different strategies, use:

  plot_load_gap_vs_balls(results)
