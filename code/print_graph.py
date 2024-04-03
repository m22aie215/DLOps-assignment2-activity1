from activation_functions import Sigmoid, ReLU, LeakyReLU, Tanh, Plotter
import numpy as np


if __name__ == "__main__":
    # Usage
    x = np.linspace(-5, 5, 100)
    activation_functions = [Sigmoid(), ReLU(), LeakyReLU(), Tanh()]
    plotter = Plotter(activation_functions, x)
    plotter.plot()
