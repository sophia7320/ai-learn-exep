class Model:
    def __init__(self):
        self.training = False

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def parameters(self):
        raise []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
