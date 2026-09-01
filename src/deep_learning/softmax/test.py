import random

import numpy as np
import torch

from src.util import data_loader

from .model import Softmax_classify

device = torch.device("cuda")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 在程序最开始调用
set_seed(42)  # 42 是常用的幸运数字，你可以换成任意整数


def accuracy(data_test, model: Softmax_classify):
    total, correct = 0, 0

    for X, y in data_test:
        pred = model.predict(X)

        total += len(y)
        correct += (pred == y).sum().item()
    return correct / total


def test():
    data_train, data_test = data_loader.load_data("fashionmnist", 256)

    model = Softmax_classify(28 * 28, 10, lr=0.005)
    # model.fit(data_train, epochs=20001, print_every=500)
    model.fit(data_train, epochs=31, print_every=10)

    print(f"test : accuracy : {accuracy(data_test, model)}")


if __name__ == "__main__":
    test()
