import os
import shutil


def delete_folders(folder_paths):
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"The folder {folder_path} has been deleted.")
        else:
            print(f"The folder {folder_path} does not exist.")


if __name__ == "__main__":
    base_dir = "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/5-experiment_logs"
    event_paths = []
    to_be_deleted_folders = [
        os.path.join(base_dir, event_path) for event_path in event_paths
    ]

    delete_folders(to_be_deleted_folders)

    print("\nOK. All folders have been deleted.")
