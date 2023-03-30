import numpy as np
from sklearn.neural_network import MLPRegressor


class NN3Wrapper():
    def __init__(self, model_params=None):
        self.model_name = "nn3"
        self.search_type = 'random'
        self.param_grid = {"early_stopping": [True],
                           "learning_rate": ["invscaling"],
                           "learning_rate_init": np.linspace(0.001, 0.999, 100),
                           'alpha': np.linspace(0.001, 0.999, 100),
                           'solver': ["adam"],
                           'activation': ["relu"],
                           "hidden_layer_sizes": [(32, 16, 8)]}
        if model_params is None:
            self.ModelClass = MLPRegressor()
        else:
            self.ModelClass = MLPRegressor(**model_params)