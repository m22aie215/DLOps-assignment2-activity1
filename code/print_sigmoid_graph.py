from activation_functions import Sigmoid, Plotter
import numpy as np


if __name__ == "__main__":
    # Usage
    random_values = np.array(
        sorted([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])
    )
    activation_functions = [Sigmoid()]
    plotter = Plotter(activation_functions, random_values)
    plotter.plot()
