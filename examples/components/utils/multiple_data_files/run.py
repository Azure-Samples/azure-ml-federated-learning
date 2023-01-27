"""This script multiplies a folder by copying the files and adding suffixes."""
import os
import argparse
import logging
import glob
import pathlib
import mlflow
import shutil
from tqdm import tqdm


def build_arguments_parser(parser: argparse.ArgumentParser = None):
    """Build the argument parser for CLI settings."""
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to folder containing some files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write multiplied dataset",
    )

    parser.add_argument(
        "--multiply",
        type=int,
        required=True,
        help="How much to multiply files.",
    )

    return parser


def multiply_files(file_paths, source, target, multiplication_factor):
    """Multiply a given folder."""
    files_created = 0

    for i in range(multiplication_factor):
        for entry in tqdm(file_paths):
            if os.path.isfile(entry):
                # get name of file
                file_name = pathlib.Path(os.path.basename(entry))

                # get path to the file
                full_input_dir = os.path.dirname(entry)

                # create path to the output
                rel_dir = os.path.relpath(full_input_dir, source)
                full_output_dir = os.path.join(target, rel_dir)

                # create a name for the output file
                output_file_path = os.path.join(
                    full_output_dir, file_name.stem + f"_{i}" + file_name.suffix
                )

                # create output dir
                if not os.path.isdir(full_output_dir):
                    logging.getLogger(__name__).info(
                        f"Creating output subfolder {full_output_dir}"
                    )
                    os.makedirs(full_output_dir, exist_ok=True)

                if not os.path.isfile(output_file_path):
                    shutil.copy(entry, output_file_path)
                    files_created += 1

        logging.getLogger(__name__).info(f"Achieved multication {i}")

    return files_created


def run(args):
    """Run the script using CLI arguments."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running with arguments: {args}")

    # MLFLOW: initialize mlflow (once in entire script)
    mlflow.start_run()

    all_files_list = list(
        glob.glob(os.path.join(args.input, "**", "*"), recursive=True)
    )

    logger.info(f"Total file list len={len(all_files_list)}")

    mlflow.log_metric("total_files_count", len(all_files_list))

    files_created = multiply_files(
        all_files_list, args.input, args.output, args.multiply
    )
    mlflow.log_metric("files_created", files_created)

    mlflow.end_run()

    logger.info("run() completed")


def main(cli_args=None):
    """Parse cli arguments and run script."""
    # initialize root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # create argument parser
    parser = build_arguments_parser()

    # runs on cli arguments
    args = parser.parse_args(cli_args)  # if None, runs on sys.argv

    # run the run function
    run(args)


if __name__ == "__main__":
    main()
