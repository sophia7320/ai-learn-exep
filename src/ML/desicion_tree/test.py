from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from .randomforest import RandomForest


def test():
    X, y = load_iris(return_X_y=True)
    # print(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("=======sklearn random forest===============")
    print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
    print(f"Feature importances: {rf.feature_importances_}")
    print()

    rf = RandomForest()
    rf.fit(X_train, y_train)
    print("=======my random forest===============")
    print(f"Accuracy: {rf.score(X_test, y_test):.4f}")
    print(f"Feature importances: {rf.feature_importances_}")
    print()


if __name__ == "__main__":
    test()
