from math import log
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


class TBLogExporter:

    def __init__(self,
                 tb_log_path,
                 custom_suffix,
                 event_prefix="events.out.tfevents."):
        self.tb_log_path = tb_log_path
        self.event_path = os.path.join(self.tb_log_path, 'events')
        self.custom_suffix = custom_suffix
        self.event_prefix = event_prefix
        self.log_file_name, self.experiment_number = self.find_latest_log_event(
        )

    def find_latest_log_event(self):

        log_event_files = [
            f for f in os.listdir(self.event_path) if
            f.endswith(self.custom_suffix) and f.startswith(self.event_prefix)
        ]
        if not log_event_files:
            raise FileNotFoundError(
                f"No event files with suffix '{self.custom_suffix}' found in {self.tb_log_path}"
            )

        experiment_number = len(log_event_files)
        return log_event_files[-1], experiment_number

    def load_tb_events(self, log_file_name):
        log_file_path = os.path.join(self.event_path, log_file_name)
        tb_event = event_accumulator.EventAccumulator(log_file_path)
        tb_event.Reload()
        return tb_event

    def plot_metrics(self, tb_event, grouped_metrics, fig_name):
        # 计算子图数量
        num_subplots = len(grouped_metrics)

        # 创建子图
        fig, axes = plt.subplots(num_subplots,
                                 1,
                                 figsize=(10, 5 * num_subplots),
                                 squeeze=False)
        axes = axes.flatten()  # Flatten the axes array for easy indexing

        # 绘制每个分组的指标
        for i, (group_name, metrics) in enumerate(grouped_metrics.items()):
            ax = axes[i]
            for metric in metrics:
                # 提取图例名称，即metric的子标签
                legend_name = metric.split('/')[-1]  # 例如 'train' 或 'val'

                data = tb_event.scalars.Items(metric)

                ax.plot([item.step for item in data],
                        [item.value for item in data],
                        label=metric)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(group_name)
            ax.legend()
            ax.set_title(group_name)

        arch, _, remaining_suffix = self.custom_suffix.partition('-')
        plt.tight_layout()
        # 构造保存路径，包含 arch 子目录
        save_path = os.path.join(
            self.tb_log_path, 'figs', arch,
            f'{remaining_suffix}-exp{self.experiment_number}-{fig_name}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        print('tb log exported successfully to', save_path)

    def export(self, grouped_metrics, fig_name):
        tb_event = self.load_tb_events(self.log_file_name)
        self.plot_metrics(tb_event, grouped_metrics, fig_name)


if __name__ == '__main__':
    # Testing

    custom_suffix = 'resnet101-batch:2048-lr:0.05-SGD'
    fig_name = 'metrics.png'
    tb_log_path = '/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/tb_logs/'

    # 实例化 TBLogExporter 类
    exporter = TBLogExporter(tb_log_path=tb_log_path,
                             custom_suffix=custom_suffix)

    # 定义想要绘制的指标
    grouped_metrics = {
        'Loss': ['Loss/train', 'Loss/val'],
        'Top1': ['Top1/train', 'Top1/val'],
        'Learning Rate': ['Learning_Rate'],
    }

    # 调用 export 方法，传入指标和图表文件名
    exporter.export(grouped_metrics=grouped_metrics, fig_name=fig_name)
