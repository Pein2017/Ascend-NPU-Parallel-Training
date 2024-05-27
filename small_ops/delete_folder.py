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
    event_paths = [
        "lr-1e-1/batch_size-1024/patience-30/2024-05-24||15-52-12",
        "lr-8e-1/batch_size-1024/patience-30/2024-05-25||03-53-45",
        "lr-8e-2/batch_size-1024/patience-30/2024-05-24||16-23-46",
        # "lr-1/batch_size-8192/patience-20/2024-05-21_13-09-48",
    ]
    to_be_deleted_folders = [
        os.path.join(base_dir, event_path) for event_path in event_paths
    ]

    delete_folders(to_be_deleted_folders)

    print("\nOK. All folders have been deleted.")
