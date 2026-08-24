from .neuron import Neuron


class Layer:
    def __init__(self, n_inputs: int, n_neurons: int, random_state=42):

        self.neurons = [
            Neuron(n_inputs, random_state=random_state) for _ in range(n_neurons)
        ]

    def __call__(self, x):
        res = [neuron(x) for neuron in self.neurons]
        return res[0] if len(res) == 1 else res

    def parameters(self):
        p = []

        for neuron in self.neurons:
            p.extend(neuron.parameters())

        return p
