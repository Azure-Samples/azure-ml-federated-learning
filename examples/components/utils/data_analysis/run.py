import argparse
import logging
import sys
import matplotlib.pyplot as plt

from pandas.api.types import is_numeric_dtype
import pandas as pd
import mlflow

SCALERS = {}


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

    parser.add_argument(
        "--training_data", type=str, required=True, help="Training CSV file"
    )
    parser.add_argument("--testing_data", type=str, required=True, help="Test CSV file")
    parser.add_argument(
        "--categorical_columns",
        type=str,
        required=False,
        help="Categorical column names separated by comma(,)",
    )
    parser.add_argument(
        "--onehot_columns_prefix",
        type=str,
        required=False,
        help="One hot encoded columns prefixes separated by comma(,), prefixes cannot contain underscores in the middle and must end with underscore",
    )
    parser.add_argument(
        "--metrics_prefix",
        type=str,
        required=False,
        help="Metrics prefix",
        default="default-prefix",
    )
    return parser


def get_corr_fig(df):
    fig, ax = plt.subplots()
    fig.set_figwidth(min(30, df.shape[1] * 2))
    corr = df.corr()
    cax = ax.matshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns.to_list(), rotation=45)
    ax.set_yticklabels(corr.columns.to_list())
    fig.colorbar(cax)
    fig.tight_layout()
    return fig


def analyze_categorical_column(df, column_name, metrics_prefix):
    value_counts = {
        str(k): str(v) for k, v in dict(df[column_name].value_counts()).items()
    }
    mlflow.log_dict(value_counts, f"{metrics_prefix}/{column_name}/value_counts.json")

    # Having figure larger than 30 inches is undesirable
    if len(value_counts) <= 60:
        fig, ax = plt.subplots()
        fig.set_figwidth(len(value_counts) / 2)
        ax = df[column_name].value_counts().plot(kind="bar")
        ax.set_xticklabels(list(value_counts), rotation=45)
        mlflow.log_figure(fig, f"{metrics_prefix}/{column_name}/density.png")
    else:
        logger.info(
            f"Skipping plotting due too many unique values for column: {column_name}"
        )


def analyze_numerical_column(df, column_name, metrics_prefix):
    column_description = {
        str(k): str(v) for k, v in dict(df[column_name].describe()).items()
    }
    mlflow.log_dict(
        column_description, f"{metrics_prefix}/{column_name}/column_description.json"
    )

    fig, ax = plt.subplots()
    ax = df[column_name].plot(
        kind="density", xlim=(df[column_name].min(), df[column_name].max())
    )
    mlflow.log_figure(fig, f"{metrics_prefix}/{column_name}/density.png")


def merge_onehot(df, onehot_columns_prefix):
    for prefix in onehot_columns_prefix:
        cols = [col for col in df.columns if col.startswith(prefix)]
        if len(cols) <= 1:
            continue

        df[prefix[:-1]] = [None] * len(df)

        for col in cols:
            val = col[len(prefix) :]
            df.loc[df[col] == 1, prefix[:-1]] = val
        df = df.drop(cols, axis=1)
    return df


def run_tabular_data_analysis(
    training_data,
    testing_data,
    categorical_columns,
    onehot_columns_prefix,
    metrics_prefix="default-prefix",
):
    """Run exploratory data analysis on training and test data.

    Args:
        training_data: Training data file that need to be processed
        testing_data: Testing data file that need to be processed
    Returns:
        None
    """

    logger.info(
        f"Raw Training Data path: {training_data}, Raw Testing Data path: {testing_data}"
    )

    # Start MLFlow run
    logger.debug(f"Setting up MLFlow...")
    # mlflow.set_experiment("exploratory-data-analysis")
    mlflow_run = mlflow.start_run()
    # root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")

    logger.debug(f"Loading data...")
    train_df = pd.read_csv(training_data)
    test_df = pd.read_csv(testing_data)

    mlflow.log_metric(
        key=f"{metrics_prefix}/train/Number of datapoints",
        value=f"{train_df.shape[0]}",
    )

    mlflow.log_metric(
        key=f"{metrics_prefix}/test/Number of datapoints",
        value=f"{test_df.shape[0]}",
    )

    # We are merging one hot encoded columns and creating categorical columns
    categorical_columns.extend([prefix[:-1] for prefix in onehot_columns_prefix])

    logger.debug(f"Train data analysis...")
    train_df = merge_onehot(train_df, onehot_columns_prefix)
    mlflow.log_figure(get_corr_fig(train_df), f"{metrics_prefix}/train/correlation.png")

    for column in train_df.columns:
        logger.info(f"Processing training dataset column: {column}")
        if column in categorical_columns:
            analyze_categorical_column(train_df, column, f"{metrics_prefix}/train")
        elif is_numeric_dtype(train_df[column]):
            analyze_numerical_column(train_df, column, f"{metrics_prefix}/train")
        else:
            logger.info(f"Skipping analysis for column: {column}")
    plt.close()

    logger.debug(f"Test data analysis...")
    test_df = merge_onehot(test_df, onehot_columns_prefix)
    mlflow.log_figure(get_corr_fig(test_df), f"{metrics_prefix}/test/correlation.png")

    for column in test_df.columns:
        logger.info(f"Processing test dataset column: {column}")
        if column in categorical_columns:
            analyze_categorical_column(test_df, column, f"{metrics_prefix}/test")
        elif is_numeric_dtype(test_df[column]):
            analyze_numerical_column(test_df, column, f"{metrics_prefix}/test")
        else:
            logger.info(f"Skipping analysis for column: {column}")
    plt.close()
    # End MLFlow run
    mlflow.end_run()


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
    logger.info(f"Running script with arguments: {args}")

    def run():
        """Run script with arguments (the core of the component).

        Args:
            args (argparse.namespace): command line arguments provided to script
        """

        run_tabular_data_analysis(
            args.training_data,
            args.testing_data,
            categorical_columns=args.categorical_columns.split(",")
            if args.categorical_columns
            else [],
            onehot_columns_prefix=args.onehot_columns_prefix.split(",")
            if args.onehot_columns_prefix
            else [],
            metrics_prefix=args.metrics_prefix,
        )

    run()


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
