import os
import sys
import argparse
import logging
import glob
import shutil
import pathlib
from distutils.util import strtobool

# local imports
from confidential_io import config_global_rsa_key, EncryptedFile


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse.

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--input_folder", type=str, required=True, help="")
    parser.add_argument("--output_folder", type=str, required=True, help="")
    parser.add_argument(
        "--method",
        type=str,
        choices=["encrypt", "decrypt", "copy"],
        required=False,
        default="copy",
    )

    # for local testing
    parser.add_argument(
        "--keyvault",
        type=str,
        required=False,
        default=None,
        help="url to the keyvault (if --method is 'encrypt')",
    )
    parser.add_argument(
        "--key_name",
        type=str,
        required=False,
        default=None,
        help="name of the key to draw for encryption (if --method is 'encrypt')",
    )

    return parser


def run(args):
    """Run the job using cli arguments provided.

    Args:
        args (argparse.Namespace): parsed arguments
    """
    if args.method == "copy":
        # unencrypted output, just use shutil copytree
        shutil.copytree(args.input_folder, args.output_folder)
    else:
        # if we're using local debug
        config_global_rsa_key(args.keyvault, args.key_name)

        # use glob to loop through all files recursively
        for entry in glob.glob(args.input_folder + "/**", recursive=True):
            if os.path.isfile(entry):
                logging.getLogger(__name__).info(f"Encrypting input file {entry}")
                # get name of file
                file_name = pathlib.Path(os.path.basename(entry))

                # get path to the file
                full_input_dir = os.path.dirname(entry)

                # create path to the output
                rel_dir = os.path.relpath(full_input_dir, args.input_folder)
                full_output_dir = os.path.join(args.output_folder, rel_dir)

                # create a name for the output file
                output_file_path = os.path.join(full_output_dir, file_name)

                # create output dir
                if not os.path.isdir(full_output_dir):
                    logging.getLogger(__name__).info(
                        f"Creating output subfolder {full_output_dir}"
                    )
                    os.makedirs(full_output_dir, exist_ok=True)

                if os.path.isfile(output_file_path):
                    logging.getLogger(__name__).warning("Overwriting existing file")

                # actually do the operations
                if args.method == "encrypt":
                    logging.getLogger(__name__).info(
                        f"Reading in clear from {entry}..."
                    )
                    with open(entry, mode="rb") as f:
                        plain_content = f.read()

                    logging.getLogger(__name__).info(
                        f"Encrypting in {output_file_path}..."
                    )
                    with EncryptedFile(output_file_path, mode="wb") as f:
                        f.write(plain_content)

                elif args.method == "decrypt":
                    logging.getLogger(__name__).info(
                        f"Reading encrypted from {entry}..."
                    )
                    with EncryptedFile(entry, mode="rb") as f:
                        plain_content = f.read()

                    logging.getLogger(__name__).info(
                        f"Writing in clear in {output_file_path}..."
                    )
                    with open(output_file_path, mode="wb") as f:
                        f.write(plain_content)


def config_logger(logger_name=__name__, level=logging.INFO):
    """Setup the logger"""
    # Set logging to sys.out
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    return logger


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # default main logger
    config_logger()
    # add logging for confidential_io
    config_logger(logger_name="confidential_io")

    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    main()
