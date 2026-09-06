import numpy as np


class DataLoader:
    def __init__(
        self, data: tuple[np.ndarray, np.ndarray], batch_size=32, rng=None, shuffle=True
    ):
        # print(data)
        self.X = data[0]
        self.y = data[1]
        self.batch_size = batch_size

        self.X_batches = np.array_split(self.X, len(self.X) // self.batch_size)
        self.y_batches = np.array_split(self.y, len(self.y) // self.batch_size)

        # print(self.X_batches, self.y_batches)

        self.rng = rng if rng is not None else np.random.default_rng(42)
        self.rng = (
            self.rng if not isinstance(self.rng, int) else np.random.default_rng(rng)
        )

        if shuffle:
            for X, y in zip(self.X_batches, self.y_batches):
                shuffle_idx = self.rng.permutation(len(X))
                X[:] = X[shuffle_idx]
                y[:] = y[shuffle_idx]

    def __iter__(self):
        yield from zip(self.X_batches, self.y_batches)

    def __len__(self):
        return len(self.X_batches)
