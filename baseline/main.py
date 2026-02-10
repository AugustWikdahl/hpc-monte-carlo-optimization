import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from memory_profiler import profile

@profile
def mc_price_option(S0, K, r, sigma, T, M, I):
    """ Function to calculate the price of a European call option using Monte Carlo pricing method.
    Arguments:
        S0: [float] initial stock/index level
        K: [float] option strike price
        r: [float] constant risk-free short rate
        sigma: [float] constant volatility
        T: [float] time to maturity in years
        M: [int] number of time steps
        I: [int] number of paths
    Returns:
        C0: float estimated option value
    """

    dt = T / M                  # delta t (time step)
    S = np.zeros((M + 1, I))
    S[0] = S0

    # generate random monte carlo paths
    for t in range(1, M + 1):
        eps = np.random.standard_normal(I)
        # simulate one step monte carlo paths
        S[t] = S[t - 1] * np.exp(
            (r - 0.5 * sigma ** 2) * dt + sigma * eps * np.sqrt(dt)
            )
    
    # disounted value of the expected payoff
    C0 = np.exp(-r * T) * np.sum(np.maximum(S[-1] - K, 0)) / I
    return C0, S

def plot_mc_paths(S, K, S0):
    final_prices = S[-1]
    # plt.style.use('seaborn-darkgrid')
    
        # Create main figure and axis
    fig = plt.figure(figsize=(16, 6))
    
    # Create main axis for price paths
    ax_paths = plt.gca()
    
    # Plot price paths
    ax_paths.plot(S[:, ::20])
    ax_paths.grid(True)
    ax_paths.set_xlabel('Time Steps')
    ax_paths.set_ylabel('Index Level')
    ax_paths.axhline(K, color='r', linestyle='dashed', linewidth=2, label='strike')
    ax_paths.scatter(0, S0, c='r', marker='x')
    ax_paths.set_title('Monte Carlo Simulation of Geometric Brownian Motion')
    ax_paths.legend()

    # Create divider for additional axes
    divider = make_axes_locatable(ax_paths)
    
    # Create new axes for distribution plot on the right
    ax_dist = divider.append_axes("right", size="24%", pad=0.1)
    
    # Separate the final prices into above and below strike
    prices_above = final_prices[final_prices >= K]
    prices_below = final_prices[final_prices < K]
    
    # Calculate bins for consistent histogram
    # bins = np.linspace(min(final_prices), max(final_prices), 50)
    
    # Create histogram
    n, bins, patches = ax_dist.hist(final_prices, bins=200, orientation='horizontal', 
                                  density=True, color='gray', alpha=0.6)
    
    # Color the bars based on bin centers
    for i in range(len(patches)):
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center >= K:
            patches[i].set_facecolor('blue')
            patches[i].set_alpha(0.6)
            if i == 0 or i == len(patches) - 1:
                patches[i].set_label('Above strike (gets discounted)')
        else:
            patches[i].set_facecolor('red')
            patches[i].set_alpha(0.6)
            if i == 0 or i == len(patches) - 1:
                patches[i].set_label('Below strike (discounted to zero)')

    
    ax_dist.set_ylim(ax_paths.get_ylim())  # Match y-axis limits
    ax_dist.set_xlabel('Final Price Distribution')
    ax_dist.grid(True)
    ax_dist.legend()
    
    # Remove y-axis labels from distribution plot
    ax_dist.set_yticklabels([])

    plt.tight_layout()
    plt.savefig('gbm_with_dist.png', bbox_inches='tight')
    plt.show()

def main():
    
    S0 = 100.0
    K = 105.0
    r = 0.05
    sigma = 0.12
    T = 0.5 # 6 months
    M = 1000
    I = 50_000

    C0, S = mc_price_option(S0, K, r, sigma, T, M, I)
    print(f">> Initial Stock Price: {S0}")
    print("="*28)
    print(f">> European Option with {T*12} months to maturity Value: {C0}")
    print("="*28)

    plot_mc_paths(S, K, S0)

if __name__ == "__main__":
    main()