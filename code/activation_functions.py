import numpy as np
import matplotlib.pyplot as plt


class ActivationFunction:
    def compute(self, x):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def compute(self, x):
        return 1 / (1 + np.exp(-x))


class ReLU(ActivationFunction):
    def compute(self, x):
        return np.maximum(0, x)


class LeakyReLU(ActivationFunction):
    def compute(self, x, alpha=0.01):
        return np.where(x > 0, x, x * alpha)


class Tanh(ActivationFunction):
    def compute(self, x):
        return np.tanh(x)


class Plotter:
    def __init__(self, activation_functions, x_range):
        self.activation_functions = activation_functions
        self.x_range = x_range

    def plot(self):
        plt.figure(figsize=(10, 8))
        for i, func in enumerate(self.activation_functions, start=1):
            y_values = func.compute(self.x_range)
            plt.subplot(2, 2, i)
            plt.plot(self.x_range, y_values, label=func.__class__.__name__)
            plt.title(f"{func.__class__.__name__} Activation Function")
            plt.grid(True)
        plt.tight_layout()
        plt.show()


# this piece of code is just for testing the function and its implementations
if __name__ == "__main__":
    # Usage
    x = np.linspace(-5, 5, 100)
    activation_functions = [Sigmoid(), ReLU(), LeakyReLU(), Tanh()]
    plotter = Plotter(activation_functions, x)
    plotter.plot()
