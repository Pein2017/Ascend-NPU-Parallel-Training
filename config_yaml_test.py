import yaml
import os

from .config import manually_get_parse, configured_parser as configured_parser_old


def load_config_from_yaml():
    # 获取当前脚本文件的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建config.yaml文件的绝对路径
    yaml_file_path = os.path.join(script_dir, 'config.yaml')
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


config_from_yaml = load_config_from_yaml()
print('---' * 5)
print(len(config_from_yaml))


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


flat_config = flatten_dict(config_from_yaml)

configured_parser = manually_get_parse(flat_config)

print(len(configured_parser._actions))
print(len(configured_parser_old._actions))


def get_parser_options(parser):
    return {
        option
        for action in parser._actions
        for option in action.option_strings if option
    }


options_configured = get_parser_options(configured_parser)
options_configured_old = get_parser_options(configured_parser_old)

added_options = options_configured - options_configured_old
removed_options = options_configured_old - options_configured

print("Added options:", added_options)
print("Removed options:", removed_options)
