import os
from collections import OrderedDict

import yaml


def load_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data, file_path):
    with open(file_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False, Dumper=CustomDumper)


def reorder_dict(data, reference):
    """
    Reorder the keys of `data` to match the order of `reference`.
    Only reorders the keys present in both `data` and `reference`.
    """
    if not isinstance(data, dict):
        return data
    ordered_data = OrderedDict()
    for key in reference:
        if key in data:
            if isinstance(data[key], dict) and isinstance(reference[key], dict):
                ordered_data[key] = reorder_dict(data[key], reference[key])
            else:
                ordered_data[key] = data[key]
    return ordered_data


def format_yaml_files(reference_yaml_path, folder_to_be_formatted):
    reference_data = load_yaml(reference_yaml_path)

    for root, _, files in os.walk(folder_to_be_formatted):
        for file in files:
            if file.endswith(".yaml"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                data = load_yaml(file_path)

                # Reorder the data to match the reference structure
                formatted_data = reorder_dict(data, reference_data)

                # Check for missing keys
                missing_keys = set(reference_data.keys()) - set(data.keys())
                if missing_keys:
                    print(f"Missing keys in {file_path}: {missing_keys}")

                # Save the formatted data back to the file
                save_yaml(formatted_data, file_path)
                print(f"Formatted {file_path} successfully.")


class CustomDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super(CustomDumper, self).increase_indent(flow, False)

    def write_line_break(self, data=None):
        super().write_line_break(data)
        if len(self.indents) == 1:  # Only add extra line breaks between top-level keys
            super().write_line_break()

    def represent_list(self, data):
        if len(data) > 0 and isinstance(data[0], (int, float, str)):
            return self.represent_sequence(
                "tag:yaml.org,2002:seq", data, flow_style=True
            )
        return super(CustomDumper, self).represent_list(data)


# Add representer for OrderedDict
def dict_representer(dumper, data):
    return dumper.represent_dict(data.items())


CustomDumper.add_representer(OrderedDict, dict_representer)
CustomDumper.add_representer(list, CustomDumper.represent_list)

if __name__ == "__main__":
    reference_yaml_path = (
        "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/config.yaml"
    )
    folder_to_be_formatted = (
        "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/1-src/yamls/debug"
    )

    format_yaml_files(reference_yaml_path, folder_to_be_formatted)
