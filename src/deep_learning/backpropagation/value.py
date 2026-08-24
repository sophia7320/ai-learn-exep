import numpy as np


class Value:
    def __init__(self, data, children=(), op: str = ""):
        self.data = data
        self.grad = 0.0

        self._children = set(children)
        self._backward = lambda: None
        self.op = op

    def __repr__(self):
        return f"Value(data = {self.data} , grad = {self.grad} )"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        x = np.clip(self.data, -500, 500)
        res = 1.0 / (1.0 + np.exp(-x))

        out = Value(res, (self,), "sigmoid")

        def _backward():
            self.grad += res * (1 - res) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


if __name__ == "__main__":
    a = Value(2)
    b = Value(3)

    res = a * a + a * b

    res.backward()

    print(f"a.g = {a.grad} , b.g = {b.grad}")
