import pandas as pd


def round_csv_columns(file_path, columns, digits):
    """
    Rounds the specified columns in a CSV file to a given number of decimal places.

    Parameters:
    file_path (str): The path to the CSV file.
    columns (list): The list of column names to be rounded.
    digits (int): The number of decimal places to round to.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Round the values in the specified columns to the given number of decimal places
    df[columns] = df[columns].apply(lambda x: round(x, digits))

    # Save the updated DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

    print(f"Values in columns {columns} have been rounded to {digits} decimal places.")


if __name__ == "__main__":
    file_path = (
        "/data/Pein/Pytorch/Ascend-NPU-Parallel-Training/3-tb_logs/commit_log.csv"
    )
    columns_to_round = ["best_train_acc1", "best_val_acc1", "best_test_acc1"]
    digits = 4

    round_csv_columns(file_path, columns_to_round, digits)
