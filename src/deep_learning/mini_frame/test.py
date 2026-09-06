import numpy as np

from . import nn, optim
from .daraloader import DataLoader
from .loss_fn import BCELoss
from .schedule import Cosine

rng = np.random.default_rng(42)


def make_circle_data(n=500):
    X = rng.uniform(-2, 2, (n, 2))
    y = X[:, 0] * X[:, 0] + X[:, 1] * X[:, 1] < 1.5
    return X, y


def train():

    model = nn.Sequential(
        nn.Linear(2, 16, rng=rng),
        nn.GeLU(),
        nn.Linear(16, 16, rng=rng),
        nn.GeLU(),
        nn.Linear(16, 8, rng=rng),
        nn.GeLU(),
        nn.Linear(8, 1, rng=rng),
        nn.Sigmoid(),
    )

    # criterion = BCELoss(reduction="sum")
    criterion = BCELoss(reduction="sum")
    optimizer = optim.AdamW(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.05)

    scheduler = Cosine(optimizer, total_steps=20000)

    X, y = make_circle_data(500)
    split = int(len(y) * 0.8)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    loader = DataLoader((X_train, y_train), batch_size=16, shuffle=True)

    model.train()

    for epoch in range(401):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for batch_inputs, batch_targets in loader:
            # print(batch_inputs, batch_targets)

            y_pred = model.forward(batch_inputs)
            batch_loss = criterion(y_pred, batch_targets)
            grad = criterion.backward()
            # print(grad.shape)
            model.backward(grad)
            optimizer.step()

            # if epoch % 99 == 0:
            #     print(f"total_correct : {total_correct}")
            total_correct += np.sum((y_pred >= 0.5).flatten() == batch_targets)
            total_samples += batch_targets.shape[0]

            total_loss += batch_loss

        scheduler.step()
        # print(f"total loss : {total_loss:.2f}")
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples * 100

        if epoch % 40 == 0 or epoch == 99:
            # print(
            #     f"Epoch {epoch} | total correct: {total_correct} | total samples: {total_samples} | accuracy: {accuracy:.2f}% | loss: {avg_loss:.2f}"
            # )
            print(
                f"Epoch {epoch:3d} | Loss: {avg_loss:.6f} | Train Accuracy: {accuracy:.1f}%"
            )

    # print(optimizer.parameters)

    model.eval()
    correct = 0

    y_pred = model.forward(X_test)
    correct = np.sum((y_pred >= 0.5).flatten() == y_test)

    test_accuracy = correct / len(y_test) * 100
    print(f"\nTest Accuracy: {test_accuracy:.1f}% ({correct}/{len(y_test)})")

    return model, test_accuracy


if __name__ == "__main__":
    train()
