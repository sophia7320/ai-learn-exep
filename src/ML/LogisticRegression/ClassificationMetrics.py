import numpy as np


class ClassificationMetrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true, y_pred = y_true.astype(np.bool), y_pred.astype(np.bool)

        self.tp = (y_pred & y_true).astype(int).sum()
        self.tn = (~y_pred & ~y_true).astype(int).sum()
        self.fp = (y_pred & ~y_true).astype(int).sum()
        self.fn = (~y_pred & y_true).astype(int).sum()

    def accuracy(self):
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total != 0 else 0

    def precision(self):
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0

    def recall(self):
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0

    def f1(self):
        p, r = self.precision(), self.recall()

        return 2 * p * r / (p + r) if (p + r) > 0 else 0

    def print_confusion_matrix(self):
        print("\n  Confusion Matrix:")
        print("                  Predicted")
        print("                  Pos   Neg")
        print(f"  Actual Pos     {self.tp:4d}  {self.fn:4d}")
        print(f"  Actual Neg     {self.fp:4d}  {self.tn:4d}")

    def print_report(self):
        self.print_confusion_matrix()
        print(f"\n  Accuracy:  {self.accuracy():.4f}")
        print(f"  Precision: {self.precision():.4f}")
        print(f"  Recall:    {self.recall():.4f}")
        print(f"  F1 Score:  {self.f1():.4f}")


if __name__ == "__main__":
    a = np.array([1, 1, 1, 0, 0]).astype(np.bool)
    b = np.array([0, 1, 1, 0, 1]).astype(np.bool)

    print(~a)
    print(~b)

    test = ClassificationMetrics(a, b)
    test.print_confusion_matrix()
    test.print_report()
