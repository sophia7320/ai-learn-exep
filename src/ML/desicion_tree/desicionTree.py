import numpy as np

rng = np.random.default_rng(42)


def gini_impurity(labels: np.ndarray):
    labels = np.atleast_1d(labels)
    if labels.shape[0] == 0:
        return 0

    _, uq_count = np.unique(labels, return_counts=True)
    uq_p = uq_count / uq_count.sum()
    return 1 - uq_p @ uq_p


def entropy(labels: np.ndarray):
    labels = np.atleast_1d(labels)
    if labels.shape[0] == 0:
        return 0

    _, uq_count = np.unique(labels, return_counts=True)
    uq_p = uq_count / uq_count.sum()
    return -(np.log2(uq_p) @ uq_p)


def information_gain(
    parent_labels: np.ndarray,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
    criterion="gini",
):
    measure = gini_impurity if criterion == "gini" else entropy
    n_parent, n_right, n_left = (
        parent_labels.shape[0],
        right_labels.shape[0],
        left_labels.shape[0],
    )

    if n_right == 0 or n_left == 0:
        return 0

    parent_impurity = measure(parent_labels)
    children_impurity = np.array([measure(left_labels), measure(right_labels)])

    return (
        parent_impurity
        - np.array([n_left / n_parent, n_right / n_parent]) @ children_impurity
    )


class DecisionTree:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        criterion="gini",
        max_features=None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features

        self.trees = None
        self.feature_importances = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_samples, self.n_features = X.shape[0], X.shape[1]
        self.feature_importances = np.zeros(self.n_features, dtype=np.float64)

        self.tree = self._build(X, y, depth=0)

        total = self.feature_importances.sum()
        if total > 0:
            self.feature_importances /= self.feature_importances.sum()

        return self

    def predict(self, X: np.ndarray):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def _build(self, X: np.ndarray, y: np.ndarray, depth):
        if y.shape[0] == 1:
            return {"leaf": True, "value": y[0]}

        if self.max_depth is not None and depth >= self.max_depth:
            return self._make_leaf(y)

        if y.shape[0] <= self.min_samples_split:
            return self._make_leaf(y)

        best_feature, best_threshold, best_gain = self._best_split(X, y)

        if best_feature is None or best_gain == 0:
            return self._make_leaf(y)

        # split the data with best_threshold
        mask = X[:, best_feature] <= best_threshold
        X_left, y_left = X[mask], y[mask]
        X_right, y_right = X[~mask], y[~mask]

        if (
            y_left.shape[0] < self.min_samples_leaf
            or y_right.shape[0] < self.min_samples_leaf
        ):
            return self._make_leaf(y)

        self.feature_importances[best_feature] += y.shape[0] * best_gain

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build(X_left, y_left, depth + 1),
            "right": self._build(X_right, y_right, depth + 1),
        }

    def _make_leaf(self, y: np.ndarray):
        uq, uq_count = np.unique(y, return_counts=True)
        return {"leaf": True, "value": uq[np.argmax(uq_count)]}

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gained = -1.0

        if isinstance(self.max_features, int):
            if self.max_features < 0:
                raise ValueError("the max_feature can't be nagetive!!")
            feature_indices = rng.permutation(self.n_features)[: self.max_features]
        elif callable(self.max_features):
            maxf = self.max_features(self.n_features)
            feature_indices = rng.permutation(self.n_features)[:maxf]
        else:
            feature_indices = rng.permutation(self.n_features)

        for feature in feature_indices:
            X_col = X[:, feature]
            X_uq = np.unique(X_col)
            for i in range(X_uq.shape[0] - 1):
                threshold = (X_uq[i] + X_uq[i + 1]) / 2

                mask = X_col <= threshold
                y_left = y[mask]
                y_right = y[~mask]

                if (
                    y_right.shape[0] < self.min_samples_leaf
                    or y_left.shape[0] < self.min_samples_leaf
                ):
                    continue

                gained = information_gain(y, y_left, y_right, self.criterion)

                if gained > best_gained:
                    best_feature = feature
                    best_threshold = threshold
                    best_gained = gained

        return best_feature, best_threshold, best_gained

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["value"]
        else:
            if x[node["feature"]] <= node["threshold"]:
                return self._predict_one(x, node["left"])
            else:
                return self._predict_one(x, node["right"])
