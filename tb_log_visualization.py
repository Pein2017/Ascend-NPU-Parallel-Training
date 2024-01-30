import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


def export_tb_log_to_figure(custom_suffix, fig_name, log_dir):
    """
    从 TensorBoard 日志导出数据并保存为图表。

    :param custom_suffix: 用于识别特定 TensorBoard 日志文件的自定义后缀。
    :param fig_name: 保存的图表文件名。
    :param log_dir: 存放 TensorBoard 日志文件的目录。
    """
    # 寻找匹配自定义后缀的日志文件
    log_event_files = [f for f in os.listdir(log_dir) if custom_suffix in f]
    if log_event_files:
        log_event_name = log_event_files[0]  # 假设只有一个匹配的文件
    else:
        raise FileNotFoundError(
            f"No event files with suffix '{custom_suffix}' found in {log_dir}")

    log_path = os.path.join(log_dir, log_event_name)
    tb_event = event_accumulator.EventAccumulator(log_path)
    tb_event.Reload()

    # 创建绘图和子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # 绘制 Loss（训练和验证）
    loss_train = tb_event.scalars.Items('Loss/train')
    loss_val = tb_event.scalars.Items('Loss/val')
    ax1.plot([i.step for i in loss_train], [i.value for i in loss_train],
             label='Train',
             color='blue')
    ax1.plot([i.step for i in loss_val], [i.value for i in loss_val],
             label='Val',
             color='red')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss")

    # 绘制 Top1 准确率（训练和验证）
    top1_train = tb_event.scalars.Items('Top1/train')
    top1_val = tb_event.scalars.Items('Top1/val')
    ax2.plot([i.step for i in top1_train], [i.value for i in top1_train],
             label='Train',
             color='green')
    ax2.plot([i.step for i in top1_val], [i.value for i in top1_val],
             label='Val',
             color='orange')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top1 Accuracy")
    ax2.legend()
    ax2.set_title("Top1 Accuracy")

    # 调整子图布局
    plt.tight_layout()

    # 构建保存路径并保存图像
    save_path = os.path.join(log_dir, custom_suffix, fig_name)
    plt.savefig(save_path)

    # 关闭绘图窗口
    plt.close(fig)

    print('tb log exported successfully to', save_path)
