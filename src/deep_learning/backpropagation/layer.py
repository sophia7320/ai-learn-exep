from .neuron import Neuron


class Layer:
    def __init__(self, n_inputs: int, n_neurons: int):

        self.neurons = [Neuron(n_inputs) for _ in range(n_neurons)]

    def __call__(self, x):
        res = [neuron(x) for neuron in self.neurons]
        return res[0] if len(res) == 1 else res

    def parameters(self):
        p = []

        for neuron in self.neurons:
            p.extend(neuron.parameters())

        return p
