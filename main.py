import numpy as np


def main():
    rng = np.random.default_rng(42)

    arr = rng.random(6)

    print(arr)


if __name__ == "__main__":
    main()
