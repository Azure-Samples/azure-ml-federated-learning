import logging
import sys
import argparse
import os
import json
import subprocess


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

    group = parser.add_argument_group("NVFlare Provision Wrapper")
    group.add_argument(
        "-w",
        type=str,
        required=True,
        help="path to workspace folder (output)",
    )
    group.add_argument(
        "-p",
        type=str,
        required=True,
        help="path to provision yaml (input)",
    )
    group.add_argument(
        "-r",
        type=str,
        required=True,
        help="redis url",
    )

    return parser

def main():
    """Script main function."""
    print("sys.argv: {}".format(sys.argv))

    # parse the arguments
    parser = get_arg_parser()
    args, _ = parser.parse_known_args()
    print("args: {}".format(args))

    # run command line `nvflare provision -p <provision.yaml> -w <workspace>`
    print("Running nvflare provision")
    provision_command = ["nvflare", "provision", "-p", args.p, "-w", args.w]
    print("provision_command: {}".format(provision_command))
    subprocess.run(provision_command, check=True)

    print("Creating comm_config.json")
    comm_config = {
        "url": args.r
    }

    
    # look for all "local" subfolders under path args.w
    for root, dirs, files in os.walk(args.w):
        for _dir in dirs:
            print(os.path.join(root, _dir))
            # if file is a directory called "local"
            if _dir.rstrip("/").endswith("local"):
                # create a file called `comm_config.json` inside the directory
                comm_config_path = os.path.join(root, _dir, "comm_config.json")
                print("comm_config_path: {}".format(comm_config_path))
                with open(comm_config_path, "w") as f:
                    json.dump(comm_config, f)

    # in the workspace folder, under each client folder, create a file called `comm_config.json`
    # with the following content:
    # {
    #     "redis_url": <redis_url>
    # }

if __name__ == "__main__":
    main()
