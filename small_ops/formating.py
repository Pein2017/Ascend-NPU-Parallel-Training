import os
import yaml
from collections import OrderedDict


def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    return yaml.load(stream, OrderedLoader)


def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def reorder_dict(reference, data):
    new_data = OrderedDict()
    for key, value in reference.items():
        if key in data:
            if isinstance(value, dict) and isinstance(data[key], dict):
                new_data[key] = reorder_dict(value, data[key])
            else:
                new_data[key] = data[key]
        else:
            print(f"Missing key: {key}")
    return new_data


def format_yaml_files(reference_path, folder_path):
    with open(reference_path, "r") as file:
        reference = ordered_load(file, yaml.SafeLoader)

    for filename in os.listdir(folder_path):
        if filename.endswith(".yaml") or filename.endswith(".yml"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as file:
                data = ordered_load(file, yaml.SafeLoader)
            reordered_data = reorder_dict(reference, data)
            with open(file_path, "w") as file:
                ordered_dump(reordered_data, file, Dumper=yaml.SafeDumper)
            print(f"Formatted: {file_path}")


if __name__ == "__main__":
    reference_yaml_path = (
        "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/config.yaml"
    )
    folder_to_format = (
        "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/yamls/debug"
    )
    format_yaml_files(reference_yaml_path, folder_to_format)
