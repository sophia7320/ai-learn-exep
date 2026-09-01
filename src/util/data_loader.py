import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from torchvision import transforms

device = torch.device("cuda")


def get_n_loader_workers():
    return 0


def load_data(name: str, batch_size: int):
    "[fashionmnist]"
    if name == "fashionmnist":
        raw_data_train, raw_data_test = load_fashionmnist_data()

        print("finish loading")
        return data.DataLoader(
            raw_data_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=get_n_loader_workers(),
        ), data.DataLoader(
            raw_data_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=get_n_loader_workers(),
        )


def load_fashionmnist_data() -> tuple[torch.Tensor, torch.Tensor]:
    trans = transforms.ToTensor()
    print("loading data .....")
    minist_train = torchvision.datasets.FashionMNIST(
        root="./data", train=True, transform=trans, download=True
    )
    minist_test = torchvision.datasets.FashionMNIST(
        root="./data", train=False, transform=trans, download=True
    )

    print("tensoring.....")

    res = [None, None]
    for i, ds in enumerate([minist_train, minist_test]):
        X = torch.stack([x for x, _ in ds]).to(device)
        y = torch.tensor([y for _, y in ds]).to(device)

        res[i] = data.TensorDataset(X, y)
    return res[0], res[1]

    # return minist_train, minist_test


def text_labels(indices):
    """Return text labels."""
    labels = [
        "t-shirt",
        "trouser",
        "pullover",
        "dress",
        "coat",
        "sandal",
        "shirt",
        "sneaker",
        "bag",
        "ankle boot",
    ]
    return [labels[int(i)] for i in indices]


def show_img(imgs: torch.Tensor, n_rows: int, n_cols: int, titles=None, scale=1.5):

    figsize = (scale * n_cols, scale * n_rows)  # (weight ,high)
    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes: plt.Axes
    axes = axes.flatten()

    for i, (img, ax) in enumerate(zip(imgs, axes)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)

        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        if titles:
            ax.set_title(titles[i])

    return axes


if __name__ == "__main__":
    data_train, data_test = load_fashionmnist_data()

    print(data_train, "\n", data_test)
    print(len(data_train), "\t", len(data_test))
    X, y = next(iter(data.DataLoader(data_train, batch_size=18)))
    show_img(X.reshape(18, 28, 28), 2, 9, titles=text_labels(y))
    plt.show()
