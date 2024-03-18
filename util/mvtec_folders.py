import os
import shutil
from argparse import ArgumentParser


def convert_dataset_structure(root_dir):
    test_dir = os.path.join(root_dir, "test")
    defect_types = [d for d in os.listdir(test_dir) if
        os.path.isdir(os.path.join(test_dir, d)) and d != "good"]
    os.makedirs(os.path.join(test_dir, "bad"))

    for defect_type in defect_types:
        defect_dir = os.path.join(test_dir, defect_type)
        for filename in os.listdir(defect_dir):
            old_file_path = os.path.join(defect_dir, filename)
            new_file_path = os.path.join(test_dir, "bad", f"{defect_type}_{filename}")
            shutil.move(old_file_path, new_file_path)

        # Remove the now empty defect type directory
        os.rmdir(defect_dir)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True,
        help="Relative or absolute path to the root directory of the dataset"
    )
    args = parser.parse_args()
    convert_dataset_structure(args.root_dir)
