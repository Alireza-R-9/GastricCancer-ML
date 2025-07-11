import numpy as np

class AOA:
    """
    Archimedes Optimization Algorithm (AOA) implementation.

    This algorithm is a population-based metaheuristic optimization method
    inspired by Archimedes' principle, balancing exploration and exploitation
    to find the optimal solution of a given fitness function within defined bounds.
    """

    def __init__(self, fitness_func, bounds, max_iter=50, population_size=20):
        """
        Initialize the AOA optimizer.

        Parameters:
        - fitness_func: callable
            The objective function to minimize.
        - bounds: list or array-like of shape (num_params, 2)
            The lower and upper bounds for each parameter.
        - max_iter: int, optional (default=50)
            Maximum number of iterations (generations).
        - population_size: int, optional (default=20)
            Number of candidate solutions in the population.
        """
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.max_iter = max_iter
        self.population_size = population_size
        self.num_params = len(bounds)

    def initialize_population(self):
        """
        Generate the initial population uniformly at random within bounds.

        Returns:
        - np.ndarray of shape (population_size, num_params)
            Initial candidate solutions.
        """
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=(self.population_size, self.num_params))

    def run(self):
        """
        Run the AOA optimization process.

        Returns:
        - np.ndarray
            The best-found solution vector.
        """
        # Initialize population
        X = self.initialize_population()

        # Evaluate initial fitness for the population
        fitness = np.array([self.fitness_func(ind) for ind in X])
        best_idx = np.argmin(fitness)
        best = X[best_idx].copy()

        # Main optimization loop
        for t in range(1, self.max_iter + 1):
            # Transfer function decreases linearly from 1 to 0
            TF = 1 - t / self.max_iter

            # Math Optimizer Accelerated function increases linearly from 0.2 to 1
            MOA = 0.2 + t * ((1 - 0.2) / self.max_iter)

            # Random coefficients for movement update
            alpha = np.random.uniform(0, 1, (self.population_size, self.num_params))
            rand = np.random.uniform(0, 1, (self.population_size, self.num_params))

            for i in range(self.population_size):
                if rand[i].mean() > MOA:
                    # Exploration phase:
                    # Move candidate away from its current position toward random points in the search space
                    X[i] = X[i] + alpha[i] * TF * (np.random.uniform(self.bounds[:, 0], self.bounds[:, 1]) - X[i])
                else:
                    # Exploitation phase:
                    # Move candidate toward the current best solution
                    X[i] = best + alpha[i] * TF * (best - X[i])

                # Ensure candidates stay within search space bounds
                X[i] = np.clip(X[i], self.bounds[:, 0], self.bounds[:, 1])

            # Recalculate fitness after position update
            fitness = np.array([self.fitness_func(ind) for ind in X])
            best_idx = np.argmin(fitness)

            # Update best solution found so far
            if fitness[best_idx] < self.fitness_func(best):
                best = X[best_idx].copy()

        return best
