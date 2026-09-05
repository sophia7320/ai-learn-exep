import numpy as np


class Daraloader:
    def __init__(self, data: tuple[np.ndarray, np.ndarray], batch_size=32, rng=None):
        self.X = data[0]
        self.y = data[1]
        self.batch_size = batch_size

        self.X_batches = np.array_split(self.X, len(self.X) // self.batch_size)
        self.y_batches = np.array_split(self.y, len(self.y) // self.batch_size)

        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rng = rng if not isinstance(rng, int) else np.random.default_rng(rng)

        for X, y in zip(self.X_batches, self.y_batches):
            shuffle_idx = self.rng.permutation(len(X))
            self.X_batches = X[shuffle_idx]
            self.y_batches = y[shuffle_idx]

    def __iter__(self):
        yield from zip(self.X_batches, self.y_batches)

    def __len__(self):
        return len(self.X_batches)
