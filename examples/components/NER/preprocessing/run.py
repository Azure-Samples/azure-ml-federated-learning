import argparse
import logging
import sys

from transformers import AutoTokenizer
from datasets import load_from_disk, DatasetDict
import mlflow
from distutils.util import strtobool
import os
import shutil
from datasets import concatenate_datasets

tokenizer_name = None


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

    parser.add_argument("--raw_training_data", type=str, required=True, help="")
    parser.add_argument("--raw_testing_data", type=str, required=True, help="")
    parser.add_argument("--train_output", type=str, required=True, help="")
    parser.add_argument("--test_output", type=str, required=True, help="")
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    parser.add_argument(
        "--tokenizer_name", type=str, required=False, help="Tokenizer model name"
    )
    parser.add_argument(
        "--benchmark_test_all_data", type=strtobool, required=False, default=False, help="Output folder for data",
    )
    return parser


def load_dataset(train_data_dir, test_data_dir):
    """Load dataset from {train_data_dir} and {test_data_dir}

    Args:
        train_data_dir(str, optional): Training data directory path
        test_data_dir(str, optional): Testing data directory path

    Returns:
        DatasetDict: Contains train and test datasets
    """
    logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")

    df = DatasetDict()
    df["train"] = load_from_disk(train_data_dir)
    df["test"] = load_from_disk(test_data_dir)

    return df


def align_labels_with_tokens(labels, word_ids):
    """Align labels in a sentence with tokens.

    Args:
        labels (list): a list of label_id for a sentence
        word_ids (list): a list of word ids

    Returns:
        list: a list of labels after alignment
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    """Tokenize a sentence and then align labels.
    By default, it uses the 'bert-base-cased' tokenizer.

    Args:
        examples (dict): Contains tokens and tags as keys.

    Returns:
        dict: Tokenized sentences with their corresponding labels
    """
    global tokenizer_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def preprocess_data(
    raw_train_data_dir,
    raw_test_data_dir,
    train_data_dir="./",
    test_data_dir="./",
    metrics_prefix="default-prefix",
    benchmark_test_all_data=False
):
    """Preprocess the data and save the processed data to train_data_dir and test_data_dir.

    Args:
        train_data_dir (str): Train data directory where processed train data will be saved.
        test_data_dir (str): Test data directory where processed test data will be saved.
        metrics_prefix (str): prefix to be used in the metric logging. Defaults to "default-prefix".
        total_num_of_silos (int): Total number of silos as partitions will be made based on this. Defaults to 3.
        silo_num (int): Silo number/index. Defaults to 0.
    Returns:
        None
    """
    # create dirs for partial train/test output
    partial_train_path = os.path.join(train_data_dir,"partial_train")
    partial_test_path = os.path.join(test_data_dir,"partial_test")
    os.makedirs(partial_train_path, exist_ok=True)
    os.makedirs(partial_test_path, exist_ok=True)

    # load dataset
 
    partial_raw_train_path = os.path.join(raw_train_data_dir,"partial_train")
    partial_raw_test_path = os.path.join(raw_test_data_dir,"partial_test")
    # df = load_dataset(train_data_dir, test_data_dir)


    ## This is for DDP benchmarking
    multiplier = 10
    for i in range(multiplier):
        #create dataset for each multiplier in the output dir
        partial_train_each_path = os.path.join(partial_train_path,f"dataset_{i}")
        partial_test_each_path = os.path.join(partial_test_path,f"dataset_{i}")
        os.makedirs(partial_train_each_path, exist_ok=True)
        os.makedirs(partial_test_each_path, exist_ok=True)
        #move train
        shutil.copy(os.path.join(partial_raw_train_path,f"dataset_{i}.arrow" ), os.path.join(partial_train_each_path, f"dataset.arrow" ) )
        shutil.copy(os.path.join(partial_raw_train_path,f"dataset_info_{i}.json" ), os.path.join(partial_train_each_path, f"dataset_info.json" ))
        shutil.copy(os.path.join(partial_raw_train_path,f"state_{i}.json" ), os.path.join(partial_train_each_path, f"state.json" ))
        # move test
        shutil.copy(os.path.join(partial_raw_test_path,f"dataset_{i}.arrow" ), os.path.join(partial_test_each_path, f"dataset.arrow" ))
        shutil.copy(os.path.join(partial_raw_test_path,f"dataset_info_{i}.json" ),  os.path.join(partial_test_each_path, f"dataset_info.json" ))
        shutil.copy(os.path.join(partial_raw_test_path,f"state_{i}.json" ), os.path.join(partial_test_each_path, f"state.json" ))
        #concetenate datasets
        if i == 0:
            df = load_dataset(partial_train_each_path, partial_test_each_path)  
        else:
            df_tmp = load_dataset(partial_train_each_path, partial_test_each_path)
            df["train"] = concatenate_datasets([df["train"], df_tmp["train"]], axis=0)
            df["test"] = concatenate_datasets([df["test"], df_tmp["test"]], axis=0)

    # tokenize and align labels
    tokenized_datasets = df.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=df["train"].column_names,
    )

    # mlflow logging
    log_metadata(
        tokenized_datasets["train"], tokenized_datasets["test"], metrics_prefix
    )

    # save processed data
    #tokenized_datasets["train"].save_to_disk(partial_train_path)
    #tokenized_datasets["test"].save_to_disk(partial_test_path)

    # for DDP testing
    partial_combined_train_path = os.path.join(partial_train_path,f"partial_ten_fold_combined_train")
    partial_combined_test_path = os.path.join(partial_test_path,f"partial_ten_fold_combined_test")
    os.makedirs(partial_combined_train_path, exist_ok=True )
    os.makedirs(partial_combined_test_path, exist_ok=True)
    tokenized_datasets["train"].save_to_disk(partial_combined_train_path)
    tokenized_datasets["test"].save_to_disk(partial_combined_test_path)

    if benchmark_test_all_data:
        # create dirs for all train/test output
        all_train_path = os.path.join(train_data_dir,"all_train")
        all_test_path = os.path.join(test_data_dir,"all_test")
        os.makedirs(all_train_path, exist_ok=True)
        os.makedirs(all_test_path, exist_ok=True)
        # load dataset
        all_raw_train_path = os.path.join(raw_train_data_dir,"all_train")
        all_raw_test_path = os.path.join(raw_test_data_dir,"all_test")
        #df_all_data = load_dataset(all_raw_train_path,all_raw_test_path)

        #This is for DDP benchmarking
        for i in range(multiplier):
            #create dataset for each multiplier in the output dir
            all_train_each_path = os.path.join(all_train_path,f"dataset_{i}")
            all_test_each_path = os.path.join(all_test_path,f"dataset_{i}")
            os.makedirs(all_train_each_path, exist_ok=True)
            os.makedirs(all_test_each_path, exist_ok=True)
            #move train
            shutil.copy(os.path.join(all_raw_train_path,f"dataset_{i}.arrow" ), os.path.join(all_train_each_path, f"dataset.arrow" ) )
            shutil.copy(os.path.join(all_raw_train_path,f"dataset_info_{i}.json" ), os.path.join(all_train_each_path, f"dataset_info.json" ))
            shutil.copy(os.path.join(all_raw_train_path,f"state_{i}.json" ), os.path.join(all_train_each_path, f"state.json" ))
            # move test
            shutil.copy(os.path.join(all_raw_test_path,f"dataset_{i}.arrow" ), os.path.join(all_test_each_path, f"dataset.arrow" ))
            shutil.copy(os.path.join(all_raw_test_path,f"dataset_info_{i}.json" ),  os.path.join(all_test_each_path, f"dataset_info.json" ))
            shutil.copy(os.path.join(all_raw_test_path,f"state_{i}.json" ), os.path.join(all_test_each_path, f"state.json" ))
            #concetenate datasets

            if i == 0:
                df_all_data = load_dataset(all_train_each_path, all_test_each_path)  
            else:
                df_tmp = load_dataset(all_train_each_path,all_test_each_path)
                df_all_data["train"] = concatenate_datasets([df_all_data["train"], df_tmp["train"]], axis=0)
                df_all_data["test"] = concatenate_datasets([df_all_data["test"], df_tmp["test"]], axis=0)

            #df_all_data = load_dataset(all_train_each_path, all_test_each_path) if i == 0 else concatenate_datasets([df_all_data, load_dataset(all_train_each_path, all_test_each_path)], axis=0)



        # tokenize and align labels
        tokenized_datasets = df_all_data.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=df_all_data["train"].column_names,
        )
        # mlflow logging
        log_metadata(
            tokenized_datasets["train"], tokenized_datasets["test"], metrics_prefix
        )
         # save processed data
        #tokenized_datasets["train"].save_to_disk(all_train_path)
        #tokenized_datasets["test"].save_to_disk(all_test_path)

        # for DDP testing
        all_combined_train_path = os.path.join(all_train_path,f"all_ten_fold_combined_train")
        all_combined_test_path = os.path.join(all_test_path,f"all_ten_fold_combined_test")
        os.makedirs(all_combined_train_path, exist_ok=True)
        os.makedirs(all_combined_test_path, exist_ok=True)
        tokenized_datasets["train"].save_to_disk(all_combined_train_path)
        tokenized_datasets["test"].save_to_disk(all_combined_test_path)


def log_metadata(X_train, X_test, metrics_prefix):
    """Preprocess the data and save the processed data to train_data_dir and test_data_dir.

    Args:
        X_train (Dataset): Train dataset
        X_test (Dataset): Test dataset
        metrics_prefix (str): prefix to be used in the metric logging. Defaults to "default-prefix".
    Returns:
        None
    """
    with mlflow.start_run() as mlflow_run:
        # get Mlflow client
        mlflow_client = mlflow.tracking.client.MlflowClient()
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")
        logger.debug(f"Root runId: {root_run_id}")
        if root_run_id:
            mlflow_client.log_metric(
                run_id=root_run_id,
                key=f"{metrics_prefix}/Number of train datapoints",
                value=f"{X_train.shape[0]}",
            )

            mlflow_client.log_metric(
                run_id=root_run_id,
                key=f"{metrics_prefix}/Number of test datapoints",
                value=f"{X_test.shape[0]}",
            )


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """
    global tokenizer_name
    tokenizer_name = args.tokenizer_name

    preprocess_data(
        args.raw_training_data,
        args.raw_testing_data,
        args.train_output,
        args.test_output,
        args.metrics_prefix,
        args.benchmark_test_all_data
    )


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    # Set logging to sys.out
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_format = logging.Formatter("[%(asctime)s] [%(levelname)s] - %(message)s")
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)

    main()
