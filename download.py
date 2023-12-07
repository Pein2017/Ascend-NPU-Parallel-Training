import os

# 定义数据集存储路径
data_path = '/home/HW/Pein/cifar10_data/'

# 确保数据路径存在
os.makedirs(data_path, exist_ok=True)

# 导入您的 get_dataloaders 函数
from your_data_loader_module import get_dataloaders  # 替换 your_data_loader_module 为实际模块名

# 调用函数以下载数据集
# 注意：这里不关心返回的加载器，只关心触发下载过程
_ = get_dataloaders(data_path=data_path, batch_size=1, distributed=False)

# 接下来是您的主训练逻辑
# 例如：
# if __name__ == '__main__':
#     # ... 这里是您的分布式训练启动逻辑 ...
