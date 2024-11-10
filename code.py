# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:25:36 2024

@author: Fayçal
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def simulate_ball_allocation(m, n, T=10, b=1, strategy="one_choice", beta=0.5, k=1, d=2):
    """
    Simulate the allocation of balls into bins over multiple repetitions and compute the load gap.
    
    Parameters:
        m (int): Number of bins.
        n (int): Number of balls.
        T (int): Number of repetitions for averaging the gap.
        b (int): Batch size for batched processing.
        strategy (str): Allocation strategy ("one_choice", "two_choice", "one_plus_beta", "partial_info", "d_choice").
        beta (float): Probability for one-choice in the (1 + β)-choice strategy.
        k (int): Number of questions in the partial information strategy (1 or 2).
        d (int): Number of bins to choose from in the d-choice strategy.
        
    Returns:
        tuple: (average_gap, std_dev_gap) representing the average and standard deviation of gap values.
    """
    # Array to store gap values across T repetitions
    gap_results = []

    for run in range(T):
        # Initialize load for each bin as zeros
        loads = np.zeros(m, dtype=int)
        
        # Process batches of balls
        for batch in range((n + b - 1) // b):  # Ceiling division for ⌈n/b⌉
            # Process each ball in the current batch
            for _ in range(b):
                if sum(loads) >= n:  # Stop if all balls have been allocated
                    break
                
                # Choose bins based on strategy
                if strategy == "one_choice":
                    chosen_bin = random.randint(0, m - 1)
                    
                elif strategy == "two_choice":
                    # Choose two random bins and allocate to the less-loaded bin
                    bin1, bin2 = random.sample(range(m), 2)
                    chosen_bin = bin1 if loads[bin1] <= loads[bin2] else bin2
                    
                elif strategy == "one_plus_beta":
                    # Use one-choice with probability beta, else two-choice
                    if random.random() < beta:
                        chosen_bin = random.randint(0, m - 1)
                    else:
                        bin1, bin2 = random.sample(range(m), 2)
                        chosen_bin = bin1 if loads[bin1] <= loads[bin2] else bin2
                        
                elif strategy == "partial_info":
                    # Choose using limited information, e.g., check if load above median or 25%/75% quantiles
                    bin1, bin2 = random.sample(range(m), 2)
                    median_load = np.median(loads)
                    
                    if k == 1:
                        # Only ask if bins are above the median
                        if loads[bin1] < median_load and loads[bin2] >= median_load:
                            chosen_bin = bin1
                        elif loads[bin2] < median_load and loads[bin1] >= median_load:
                            chosen_bin = bin2
                        else:
                            # If both are below or above the median, pick the lower load bin or random if equal
                            chosen_bin = bin1 if loads[bin1] <= loads[bin2] else bin2
                    
                    elif k == 2:
                        # Ask if bins are in top 25% or 75%
                        load_25 = np.percentile(loads, 25)
                        load_75 = np.percentile(loads, 75)
                        
                        if loads[bin1] < median_load and loads[bin2] >= median_load:
                            chosen_bin = bin1
                        elif loads[bin2] < median_load and loads[bin1] >= median_load:
                            chosen_bin = bin2
                        else:
                            # Both bins are either above or below the median
                            if loads[bin1] < load_75 and loads[bin2] >= load_75:
                                chosen_bin = bin1
                            elif loads[bin2] < load_75 and loads[bin1] >= load_75:
                                chosen_bin = bin2
                            elif loads[bin1] < load_25 and loads[bin2] >= load_25:
                                chosen_bin = bin1
                            elif loads[bin2] < load_25 and loads[bin1] >= load_25:
                                chosen_bin = bin2
                            else:
                                # If tie or both in the same quartile, pick the one with the lower load or random if equal
                                chosen_bin = bin1 if loads[bin1] <= loads[bin2] else bin2

                elif strategy == "d_choice":
                    # Choose d random bins and assign to the one with the least load
                    selected_bins = random.sample(range(m), d)
                    chosen_bin = min(selected_bins, key=lambda bin_idx: loads[bin_idx])

                else:
                    raise ValueError("Invalid strategy provided.")

                # Place the ball in the chosen bin
                loads[chosen_bin] += 1

        # Calculate maximum load, and gap for this repetition
        max_load = max(loads)
        gap = max_load - n/m
        gap_results.append(gap)

    # Compute average and standard deviation of gaps over T repetitions
    average_gap = np.mean(gap_results)
    std_dev_gap = np.std(gap_results)

    return average_gap, std_dev_gap


def run_allocation_experiments(m_values = [100], n_factors = [0.25, 0.5, 1, 5, 10, 20, 50, 'm_squared'], T = 10, b = 1, beta_values = [0.75, 0.5, 0.25], k_values = [1, 2], d_values = [5, 10], strategies = ["one_choice", "two_choice", "one_plus_beta", "partial_info", "d_choice"]):
    

    # Dictionary to store results
    results = []

    # Loop over each value of m
    for m in m_values:
        # Compute each n based on the factors and m
        n_values = [int(factor * m) if factor != 'm_squared' else m ** 2 for factor in n_factors]

        for n in n_values:
            for strategy in strategies:
                if strategy == "one_plus_beta":
                    # For (1 + β)-choice, vary beta
                    for beta in beta_values:
                        avg_gap, std_gap = simulate_ball_allocation(m, n, T, b, strategy, beta=beta)
                        results.append({
                            "m": m, "n": n, "strategy": strategy, "beta": beta,
                            "avg_gap": avg_gap, "std_dev_gap": std_gap
                        })

                elif strategy == "partial_info":
                    # For partial information, vary k
                    for k in k_values:
                        avg_gap, std_gap = simulate_ball_allocation(m, n, T, b, strategy, k=k)
                        results.append({
                            "m": m, "n": n, "strategy": strategy, "k": k,
                            "avg_gap": avg_gap, "std_dev_gap": std_gap
                        })

                elif strategy == "d_choice":
                    # For d-choice, vary d
                    for d in d_values:
                        avg_gap, std_gap = simulate_ball_allocation(m, n, T, b, strategy, d=d)
                        results.append({
                            "m": m, "n": n, "strategy": strategy, "d": d,
                            "avg_gap": avg_gap, "std_dev_gap": std_gap
                        })

                else:
                    # For one-choice and two-choice, no additional parameters
                    avg_gap, std_gap = simulate_ball_allocation(m, n, T, b, strategy)
                    results.append({
                        "m": m, "n": n, "strategy": strategy,
                        "avg_gap": avg_gap, "std_dev_gap": std_gap
                    })

    return results


def plot_allocation_experiment_results(results):
    # Group results by strategy
    strategies = set(result["strategy"] for result in results)
    m = results[0]["m"]  # Assuming a fixed m for all experiments
    n_values = sorted(set(result["n"] for result in results))  # Get unique n values in ascending order

    # Create the plots
    for strategy in strategies:
        # Filter results for the current strategy
        strategy_results = [res for res in results if res["strategy"] == strategy]

        plt.figure(figsize=(14, 6))

        # Plot Average Gap
        plt.subplot(1, 2, 1)
        for param_set in set((res.get("beta"), res.get("k"), res.get("d")) for res in strategy_results):
            filtered = [res for res in strategy_results if (res.get("beta"), res.get("k"), res.get("d")) == param_set]
            print(filtered)
            avg_gaps = [res["avg_gap"] for res in filtered]
            label = f"{strategy} - β={param_set[0]} k={param_set[1]} d={param_set[2]}"
            plt.plot(n_values, avg_gaps, label=label, marker="o", linewidth=2)

            # Add red dot at n = m to indicate light-load scenario
            if m in n_values:
                m_index = n_values.index(m)
                plt.plot(n_values[m_index], avg_gaps[m_index], "ro", linewidth=2)

        plt.xlabel("Number of Balls (n)")
        plt.ylabel("Average Gap")
        plt.title(f"Average Gap for Strategy: {strategy}")
        plt.legend(loc="upper left")
        plt.grid(True)

        # Plot Standard Deviation of Gap
        plt.subplot(1, 2, 2)
        for param_set in set((res.get("beta"), res.get("k"), res.get("d")) for res in strategy_results):
            filtered = [res for res in strategy_results if (res.get("beta"), res.get("k"), res.get("d")) == param_set]
            std_devs = [res["std_dev_gap"] for res in filtered]
            label = f"{strategy} - β={param_set[0]} k={param_set[1]} d={param_set[2]}"
            plt.plot(n_values, std_devs, label=label, marker="o", linewidth=2)

            # Add red dot at n = m to indicate light-load scenario
            if m in n_values:
                m_index = n_values.index(m)
                plt.plot(n_values[m_index], std_devs[m_index], "ro", linewidth=2)

        plt.xlabel("Number of Balls (n)")
        plt.ylabel("Standard Deviation of Gap")
        plt.title(f"Standard Deviation of Gap for Strategy: {strategy}")
        plt.legend(loc="upper left")
        plt.grid(True)

        # Adjust layout and show
        plt.suptitle(f"Results for {strategy} Strategy")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # Create a superimposed plot showing all strategies for both average gap and standard deviation of gap
    plt.figure(figsize=(14, 6))

    # Plot Average Gap (Superimposed)
    plt.subplot(1, 2, 1)
    for strategy in strategies:
        # Filter results for the current strategy
        strategy_results = [res for res in results if res["strategy"] == strategy]

        for param_set in set((res.get("beta"), res.get("k"), res.get("d")) for res in strategy_results):
            filtered = [res for res in strategy_results if (res.get("beta"), res.get("k"), res.get("d")) == param_set]
            avg_gaps = [res["avg_gap"] for res in filtered]
            label = f"{strategy} - β={param_set[0]} k={param_set[1]} d={param_set[2]}"
            plt.plot(n_values, avg_gaps, label=label, marker="o", linewidth=2)

            # Add red dot at n = m to indicate light-load scenario
            if m in n_values:
                m_index = n_values.index(m)
                plt.plot(n_values[m_index], avg_gaps[m_index], "ro", linewidth=2)

    plt.xlabel("Number of Balls (n)")
    plt.ylabel("Average Gap")
    plt.title("Average Gap (Superimposed for all strategies)")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Plot Standard Deviation of Gap (Superimposed)
    plt.subplot(1, 2, 2)
    for strategy in strategies:
        # Filter results for the current strategy
        strategy_results = [res for res in results if res["strategy"] == strategy]

        for param_set in set((res.get("beta"), res.get("k"), res.get("d")) for res in strategy_results):
            filtered = [res for res in strategy_results if (res.get("beta"), res.get("k"), res.get("d")) == param_set]
            std_devs = [res["std_dev_gap"] for res in filtered]
            label = f"{strategy} - β={param_set[0]} k={param_set[1]} d={param_set[2]}"
            plt.plot(n_values, std_devs, label=label, marker="o", linewidth=2)

            # Add red dot at n = m to indicate light-load scenario
            if m in n_values:
                m_index = n_values.index(m)
                plt.plot(n_values[m_index], std_devs[m_index], "ro", linewidth=2)

    plt.xlabel("Number of Balls (n)")
    plt.ylabel("Standard Deviation of Gap")
    plt.title("Standard Deviation of Gap (Superimposed for all strategies)")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Adjust layout and show
    plt.suptitle("Superimposed Results for All Strategies")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

'''
# Run the experiments, store results, and plot them
experiment_results = run_allocation_experiments()
plot_allocation_experiment_results(experiment_results)

'''

def plot_avg_gap_vs_batch_size_superimposed(m, batch_sizes, strategies, T=10, beta=[0.75, 0.5, 0.25], k=[1, 2], d=[5, 10]):
    """
    Plot the average gap as a function of batch size for each strategy, all superimposed on one plot.
    
    Args:
    - m: Number of bins (fixed at 100)
    - batch_sizes: List of batch sizes (b)
    - strategies: List of strategies to evaluate
    - T: Number of repetitions
    - beta: List of beta values for (1+β)-choice strategy
    - k: List of k values for partial information strategy
    - d: List of d values for d-choice strategy
    """
    plt.figure(figsize=(12, 8))
    
    # Iterate over each strategy and plot the average gap vs batch size
    for strategy in strategies:
        all_avg_gaps = []  # To store all the average gaps for the current strategy
        
        for b in batch_sizes:
            avg_gaps = []  # To store average gaps for a given batch size
            
            if strategy == "one_plus_beta":
                for beta_val in beta:
                    gaps = []
                    for n in range(b, m**2 + 1, b):
                        avg_gap, _ = simulate_ball_allocation(m, n, T=T, b=b, strategy=strategy, beta=beta_val)
                        gaps.append(avg_gap)
                    avg_gaps.append(np.mean(gaps))
                    
            elif strategy == "partial_info":
                for k_val in k:
                    gaps = []
                    for n in range(b, m**2 + 1, b):
                        avg_gap, _ = simulate_ball_allocation(m, n, T=T, b=b, strategy=strategy, k=k_val)
                        gaps.append(avg_gap)
                    avg_gaps.append(np.mean(gaps))
                    
            elif strategy == "d_choice":
                for d_val in d:
                    gaps = []
                    for n in range(b, m**2 + 1, b):
                        avg_gap, _ = simulate_ball_allocation(m, n, T=T, b=b, strategy=strategy, d=d_val)
                        gaps.append(avg_gap)
                    avg_gaps.append(np.mean(gaps))
            
            else:  # For strategies without extra parameters
                gaps = []
                for n in range(b, m**2 + 1, b):
                    avg_gap, _ = simulate_ball_allocation(m, n, T=T, b=b, strategy=strategy)
                    gaps.append(avg_gap)
                avg_gaps.append(np.mean(gaps))
            
            all_avg_gaps.append(np.mean(avg_gaps))  # Store the mean of avg gaps for the current batch size
            
        # Plot the average gap vs batch size for this strategy
        plt.plot(batch_sizes, all_avg_gaps, label=strategy, marker="o", linewidth=2)
    

    
    # Labeling the plot
    plt.xlabel("Batch Size (b)")
    plt.ylabel("Average Gap")
    plt.title("Average Gap vs Batch Size for Different Strategies")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Example Usage

m = 10  # Fixed m
batch_sizes = [m, 2*m, 5*m, 10*m, 20*m, 50*m, 100*m]  # List of batch sizes
strategies = ['one_choice', 'two_choice', 'one_plus_beta', 'partial_info', 'd_choice']  # Different strategies

# Plot the average gap vs batch size for each strategy
plot_avg_gap_vs_batch_size_superimposed(m, batch_sizes, strategies)
