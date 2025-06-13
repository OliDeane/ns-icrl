import matplotlib.pyplot as plt
import numpy as np

def plot_nsircl(iterations, symplex_data):
    """
    Objective: plot the norm violations at each iteration - only for a single set of data

    Input:
        iterations: list of length number of iterations
        nsicrl_data: list of norm violations from the nsircl system

    """

    # Plotting the lines
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, symplex_data, marker='o', linestyle='-', label='NSIRL')

    # Labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Constraint Violations')
    # plt.title('Constraint Violations Over Iterations')
    plt.legend()

    plt.xticks(np.arange(min(iterations), max(iterations) + 1, 2))  # Adjust step size if needed
    
    # Show plot
    plt.grid(True)

    plt.savefig("figures/nsircl_norm_violations.png", dpi=300, bbox_inches='tight')  # you can change filename and dpi as needed

    plt.show()
    
def three_line_plot(iterations, symplex_data, lcql_data, icrl_data):


    # Plotting the lines
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, symplex_data, marker='o', linestyle='-', label='NSIRL')
    plt.plot(iterations, lcql_data, marker='s', linestyle='--', label='LCQL')
    plt.plot(iterations, icrl_data, marker='^', linestyle='-.', label='ME-ICRL')

    # Labels and title
    plt.xlabel('Iterations')
    plt.ylabel('Constraint Violations')
    # plt.title('Constraint Violations Over Iterations')
    plt.legend()

    plt.xticks(np.arange(min(iterations), max(iterations) + 1, 2))  # Adjust step size if needed
    
    # Show plot
    plt.grid(True)

    plt.savefig("figures/hard_constraints.png", dpi=300, bbox_inches='tight')  # you can change filename and dpi as needed

    plt.show()

