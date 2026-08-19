import numpy as np

from .decisionTree import DecisionTree, rng


class RandomForest:
    def __init__(
        self,
        n_trees=100,
        max_depth=None,
        min_samples_split=2,
        max_features="sqrt",
        criterion="gini",
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion

        self.trees: list[DecisionTree] = []
        self.feature_importances_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_sample = y.shape[0]

        for _ in range(self.n_trees):
            indices = rng.integers(0, n_sample, n_sample)
            X_boot, y_boot = X[indices], y[indices]

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion=self.criterion,
                max_features=self.max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        self.feature_importances_ = np.array(
            [tree.feature_importances for tree in self.trees]
        ).mean(axis=0)
        self.feature_importances_ /= self.feature_importances_.sum()

    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        return np.apply_along_axis(
            lambda collumn: np.bincount(collumn).argmax(), arr=all_preds, axis=0
        )

    def score(self, X_test: np.ndarray, y_test: np.ndarray):
        y_pred = self.predict(X_test)

        return np.sum(y_pred == y_test) / y_test.shape[0]
