import os

import yaml


def load_config_from_yaml():
    # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建config.yaml文件的绝对路径
    yaml_file_path = os.path.join(script_dir, "config.yaml")
    with open(yaml_file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


config_from_yaml = load_config_from_yaml()

if __name__ == "__main__":
    print(config_from_yaml)
