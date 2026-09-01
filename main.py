import torch


def main():
    # 1. 检查 PyTorch 版本
    print("PyTorch 版本:", torch.__version__)

    # 2. 检查 GPU 是否可用
    cuda_ok = torch.cuda.is_available()
    print("GPU 是否可用:", cuda_ok)

    if cuda_ok:
        print("显卡名称:", torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # 3. 创建张量并放到指定设备（CPU 或 GPU）上计算
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)
    z = x + y

    print("计算结果:", z)


if __name__ == "__main__":
    base = "https://www.bilibili.com/video/BV16jyuBBEom?spm_id_from=333.788.videopod.episodes&vd_source=15e6728bdde45c56e241526a31856a24"
    with open("urls.txt", "w") as f:
        f.write(base + "\n")
        f.writelines(base + "&p=" + str(i) + "\n" for i in range(2, 143))
