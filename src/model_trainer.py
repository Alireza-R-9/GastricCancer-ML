from sklearn.svm import SVC
from src.aoa_optimizer import AOA

def train_svm_aoa(X, y):
    """
    Train an SVM classifier with hyperparameters optimized by Archimedes Optimization Algorithm (AOA).

    Parameters:
    - X: array-like
        Feature matrix for training.
    - y: array-like
        Target labels for training.

    Returns:
    - model: sklearn.svm.SVC
        Trained SVM model with optimized hyperparameters.
    - best_params: np.ndarray
        Array containing the best found values for C and gamma.
    """

    def fitness_function(params):
        """
        Fitness function to minimize during optimization.

        Parameters:
        - params: list or array-like
            Hyperparameters [C, gamma] for SVM.

        Returns:
        - float
            Cost to minimize (1 - training accuracy).
        """
        C, gamma = params
        model = SVC(C=C, gamma=gamma, kernel='rbf')
        # Fit the model and compute training accuracy
        accuracy = model.fit(X, y).score(X, y)
        # Return cost as 1 - accuracy to minimize
        return 1 - accuracy

    # Define bounds for hyperparameters C and gamma
    bounds = [(0.1, 100), (0.0001, 1.0)]

    # Initialize AOA optimizer with fitness function and search bounds
    optimizer = AOA(fitness_function, bounds, max_iter=20, population_size=10)

    # Run optimization to find best hyperparameters
    best_params = optimizer.run()

    # Train final SVM model with optimized hyperparameters
    model = SVC(C=best_params[0], gamma=best_params[1], kernel='rbf')
    model.fit(X, y)

    return model, best_params
