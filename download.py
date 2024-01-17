import os
from data_loader import get_dataloaders  # 确保这里正确地引用了包含 get_dataloaders 函数的模块

# 获取当前脚本文件的绝对路径
current_script_path = os.path.abspath(__file__)
# 获取当前脚本所在的目录
current_script_dir = os.path.dirname(current_script_path)
# 设置数据集存储路径为该目录下的子目录 'cifar10_data'
data_path = os.path.join(current_script_dir, 'cifar10_data')
# 确保数据路径存在
os.makedirs(data_path, exist_ok=True)


# 调用函数以下载数据集
# 注意：这里不关心返回的加载器，只关心触发下载过程
_ = get_dataloaders(data_path=data_path, batch_size=1, distributed=False)

