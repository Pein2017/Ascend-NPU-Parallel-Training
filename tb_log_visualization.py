from math import log
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator


class TBLogExporter:

    def __init__(self,
                 log_dir,
                 custom_suffix,
                 event_prefix="events.out.tfevents."):
        self.log_dir = log_dir
        self.custom_suffix = custom_suffix
        self.event_prefix = event_prefix
        self.log_file_name, self.experiment_number = self.find_latest_log_file(
        )

    def find_latest_log_file(self):
        log_event_files = [
            f for f in os.listdir(self.log_dir) if
            f.endswith(self.custom_suffix) and f.startswith(self.event_prefix)
        ]
        if not log_event_files:
            raise FileNotFoundError(
                f"No event files with suffix '{self.custom_suffix}' found in {self.log_dir}"
            )

        experiment_number = len(log_event_files)
        return log_event_files[-1], experiment_number

    def load_tb_events(self, log_file_name):
        log_path = os.path.join(self.log_dir, log_file_name)
        tb_event = event_accumulator.EventAccumulator(log_path)
        tb_event.Reload()
        return tb_event

    def plot_metrics(self, tb_event, metrics, fig_name):
        grouped_metrics = {
            'Loss': ['Loss/train', 'Loss/val'],
            'Top1': ['Top1/train', 'Top1/val'],
            'Others': []
        }

        # 将非 Loss 和 Top1 的指标放入 'Others'
        for metric in metrics:
            if metric not in grouped_metrics[
                    'Loss'] and metric not in grouped_metrics['Top1']:
                grouped_metrics['Others'].append(metric)

        num_groups = len(
            [group for group in grouped_metrics.values() if group])
        fig, axes = plt.subplots(num_groups, 1, figsize=(10, 5 * num_groups))
        if num_groups == 1:
            axes = [axes]

        for i, (group_name,
                group_metrics) in enumerate(grouped_metrics.items()):
            if group_metrics:
                for metric in group_metrics:
                    data = tb_event.scalars.Items(metric)
                    axes[i].plot([item.step for item in data],
                                 [item.value for item in data],
                                 label=metric)
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(group_name)
                axes[i].legend()
                axes[i].set_title(group_name)

        plt.tight_layout()
        save_path = os.path.join(
            self.log_dir, 'fig',
            f'{self.custom_suffix}-exp{self.experiment_number}-{fig_name}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close(fig)
        print('tb log exported successfully to', save_path)

    def export(self, metrics, fig_name):
        tb_event = self.load_tb_events(self.log_file_name)
        self.plot_metrics(tb_event, metrics, fig_name)


if __name__ == '__main__':
    # Testing
    custom_suffix = 'resnet152-batch:2048-lr:0.002'
    fig_name = 'metric.png'
    tb_log_path = '/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/tb_logs'

    # 实例化 TBLogExporter 类
    exporter = TBLogExporter(log_dir=tb_log_path, custom_suffix=custom_suffix)

    # 定义想要绘制的指标
    metrics = ['Loss/train', 'Loss/val', 'Top1/train', 'Top1/val']

    # 调用 export 方法，传入指标和图表文件名
    exporter.export(metrics=metrics, fig_name=fig_name)
