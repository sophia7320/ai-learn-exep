import random

import torch
from torch import nn

from .network import Network

device = torch.device("cuda")


def xor_test():
    xor_data = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    model = Network([2, 4, 1], learning_rate=0.6, random_state=42)

    model.fit(xor_data)

    for input, lable in xor_data:
        print(f"input : {input} -> result: {model(input)} | true: {lable} ")


def generate_circle_data(n=100):
    random.seed(7)
    data = []
    for _ in range(n):
        x1 = random.uniform(-1.5, 1.5)
        x2 = random.uniform(-1.5, 1.5)
        label = 1.0 if x1 * x1 + x2 * x2 < 1.0 else 0.0
        data.append(([x1, x2], label))
    return data


def circle_test():
    circle_data = generate_circle_data(80)

    circle_net = Network([2, 8, 1], learning_rate=1)
    random.shuffle(circle_data)

    data_train = circle_data[:60]
    data_test = circle_data[60:]
    circle_net.fit(data_train)

    correct, total = 0, len(data_test)

    for x, true in data_test:
        result = circle_net(x)
        predict = 1.0 if result.data >= 0.5 else 0.0
        print(f"x:{x} -> result: {result} | true : {true}")
        if predict == true:
            correct += 1

    print(f"accuracy = {correct / total}")


def pyt_xor():
    pass


def wash(data):
    X, y = [], []

    for x, label in data:
        X.append(x)
        y.append(label)

    return X, y


def pyt_circle():
    random.seed(50)
    raw_data = generate_circle_data(2000)
    random.shuffle(raw_data)
    split = int(2000 * 0.8)

    X, y = wash(raw_data)

    X, y = (
        torch.tensor(X, device=device),
        torch.tensor(y, device=device)[:, torch.newaxis],
    )

    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = nn.Sequential(
        nn.Linear(2, 8), nn.Sigmoid(), nn.Linear(8, 1), nn.Sigmoid()
    ).to(device)

    optimizor = torch.optim.SGD(model.parameters(), lr=1)
    criterion = torch.nn.MSELoss()

    for epoch in range(20000):
        pred = model(X_train)

        loss = criterion(pred, y_train)
        optimizor.zero_grad()
        loss.backward()
        optimizor.step()
        if epoch % 200 == 0:
            print(f"loss : {loss}")

    y_pred: torch.Tensor = model(X_test)

    correct = torch.sum((y_pred >= 0.5) == y_test)

    print(f"accucary : {correct / y_test.shape[0]:.2f}")


if __name__ == "__main__":
    print("=====================xor test=========================")
    # xor_test()
    print("=====================circle test======================")
    # circle_test()
    print("=====================circle torch=====================")
    pyt_circle()
