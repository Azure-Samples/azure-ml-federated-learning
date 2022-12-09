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
        help="Categorical column names split by comma(,)",
    )
    parser.add_argument(
        "--metrics_prefix", type=str, required=False, help="Metrics prefix"
    )
    return parser


def get_corr_fig(df):
    fig, ax = plt.subplots()
    corr = df.corr()
    cax = ax.matshow(corr)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns.to_list(), rotation=45)
    ax.set_yticklabels(corr.columns.to_list())
    fig.colorbar(cax)
    return fig


def analyze_categorical_column(df, column_name, metrics_prefix):
    for k, v in dict(df[column_name].value_counts()).items():
        mlflow.log_metric(
            key=f"{metrics_prefix}/{column_name}/value_counts/{k}",
            value=v,
        )

    fig, ax = plt.subplots()
    ax = df[column_name].value_counts().plot(kind="bar")
    mlflow.log_figure(fig, f"{metrics_prefix}/{column_name}/density.png")


def analyze_numerical_column(df, column_name, metrics_prefix):
    for k, v in dict(df[column_name].describe()).items():
        k = k.replace("%", "-percentile")
        mlflow.log_metric(
            key=f"{metrics_prefix}/{column_name}/{k}",
            value=v,
        )

    fig, ax = plt.subplots()
    ax = df[column_name].plot(
        kind="density", xlim=(df[column_name].min(), df[column_name].max())
    )
    mlflow.log_figure(fig, f"{metrics_prefix}/{column_name}/density.png")


def run_tabular_data_analysis(
    training_data,
    testing_data,
    categorical_columns,
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
    mlflow.set_experiment("exploratory-data-analysis")
    mlflow_run = mlflow.start_run()
    # root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")

    logger.debug(f"Loading data...")
    train_df = pd.read_csv(training_data, index_col=0)
    test_df = pd.read_csv(testing_data, index_col=0)

    mlflow.log_metric(
        key=f"{metrics_prefix}/train/Number of datapoints",
        value=f"{train_df.shape[0]}",
    )

    mlflow.log_metric(
        key=f"{metrics_prefix}/test/Number of datapoints",
        value=f"{test_df.shape[0]}",
    )

    logger.debug(f"Train data analysis...")
    mlflow.log_figure(get_corr_fig(train_df), f"{metrics_prefix}/train/correlation.png")
    for column in train_df.columns:
        if column in categorical_columns:
            analyze_categorical_column(
                train_df, column, f"{metrics_prefix}/train"
            )
        elif is_numeric_dtype(train_df[column]):
            analyze_numerical_column(
                train_df, column, f"{metrics_prefix}/train"
            )
        else:
            logger.info(f"Skipping analysis for column: {column}")

    logger.debug(f"Test data analysis...")
    mlflow.log_figure(get_corr_fig(test_df), f"{metrics_prefix}/test/correlation.png")
    for column in test_df.columns:
        if column in categorical_columns:
            analyze_categorical_column(
                test_df, column, f"{metrics_prefix}/test"
            )
        elif is_numeric_dtype(test_df[column]):
            analyze_numerical_column(
                test_df, column, f"{metrics_prefix}/test"
            )
        else:
            logger.info(f"Skipping analysis for column: {column}")

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
            metrics_prefix=args.metrics_prefix if args.metrics_prefix else "default-prefix",
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
