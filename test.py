import torch


def test_cuda_and_cudnn():
    # 检查 CUDA 是否可用
    if not torch.cuda.is_available():
        print("CUDA 不可用，请检查 CUDA 安装或 GPU 驱动。")
        return
    
    # 检查 cuDNN 是否可用
    if not torch.backends.cudnn.is_available():
        print("cuDNN 不可用，请检查 cuDNN 安装。")
        return

    print("CUDA 和 cuDNN 正常可用！")

    # 创建一个简单的张量并将其移动到 GPU
    device = torch.device('cuda:1')
    x = torch.randn(3, 3).to(device)
    print("张量 x 已成功移动到 GPU：")
    print(x)

    # 执行一次简单的矩阵乘法
    y = torch.matmul(x, x)
    print("张量乘法结果：")
    print(y)

if __name__ == "__main__":
    test_cuda_and_cudnn()