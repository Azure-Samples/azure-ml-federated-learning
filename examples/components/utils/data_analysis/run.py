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
    parser.add_argument(
        "--silo_index",
        type=int,
        required=True,
        help="Index of the silo",
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


def log_dict(client, run_id, metrics_prefix, column_name, silo_index, key, dictionary):
    """Log mlflow metrics

    Args:
        client (MlflowClient): Mlflow Client where metrics will be logged
        run_id (str): Run ID
        key (str): metrics x-axis label
        value (float): metrics y-axis values
        pipeline_level (bool): whether the metrics is logged at the pipeline or the job level

    Returns:
        None
    """
    assert run_id is not None
    assert metrics_prefix is not None
    assert column_name is not None
    assert silo_index is not None
    assert key is not None
    assert dictionary is not None

    # with mlflow.start_run() as run:
    #     c = mlflow.MlflowClient()
    #     c.log_dict()

    client.log_dict(
        run_id=run_id,
        dictionary=dictionary,
        artifact_file=f"{metrics_prefix}/{column_name}/{silo_index}/{key}.json",
    )


def log_figure(client, run_id, metrics_prefix, column_name, silo_index, key, figure):
    """Log mlflow metrics

    Args:
        client (MlflowClient): Mlflow Client where metrics will be logged
        run_id (str): Run ID
        key (str): metrics x-axis label
        value (float): metrics y-axis values
        pipeline_level (bool): whether the metrics is logged at the pipeline or the job level

    Returns:
        None
    """
    assert run_id is not None
    assert metrics_prefix is not None
    assert column_name is not None
    assert silo_index is not None
    assert key is not None
    assert figure is not None

    client.log_figure(
        run_id=run_id,
        figure=figure,
        artifact_file=f"{metrics_prefix}/{column_name}/{silo_index}/{key}.png",
    )
    plt.close()


def log_metrics(client, run_id, metrics_prefix, column_name, silo_index, key, value):
    """Log mlflow metrics

    Args:
        client (MlflowClient): Mlflow Client where metrics will be logged
        run_id (str): Run ID
        key (str): metrics x-axis label
        value (float): metrics y-axis values
        pipeline_level (bool): whether the metrics is logged at the pipeline or the job level

    Returns:
        None
    """

    # f"iteration_{self._iteration_num}/{self._experiment_name}/{key}"
    client.log_metric(
        run_id=run_id,
        key=f"{metrics_prefix}/{column_name}/{silo_index}/{key}",
        value=value,
    )


def analyze_categorical_column(
    df, column_name, silo_index, mlflow_client, run_id, metrics_prefix
):
    value_counts = {
        str(k): str(v) for k, v in dict(df[column_name].value_counts()).items()
    }
    log_dict(
        mlflow_client,
        run_id,
        metrics_prefix,
        column_name,
        silo_index,
        "value_counts",
        value_counts,
    )

    # Having figure larger than 30 inches is undesirable
    if len(value_counts) <= 60:
        fig, ax = plt.subplots()
        fig.set_figwidth(len(value_counts) / 2)
        ax = df[column_name].value_counts().plot(kind="bar")
        ax.set_xticklabels(list(value_counts), rotation=45)
        log_figure(
            mlflow_client,
            run_id,
            metrics_prefix,
            column_name,
            silo_index,
            "density",
            fig,
        )
    else:
        logger.info(
            f"Skipping plotting due too many unique values for column: {column_name}"
        )


def analyze_numerical_column(
    df, column_name, silo_index, mlflow_client, run_id, metrics_prefix
):
    column_description = {
        str(k): str(v) for k, v in dict(df[column_name].describe()).items()
    }
    log_dict(
        mlflow_client,
        run_id,
        metrics_prefix,
        column_name,
        silo_index,
        "column_description",
        column_description,
    )

    fig, ax = plt.subplots()
    ax = df[column_name].plot(
        kind="density", xlim=(df[column_name].min(), df[column_name].max())
    )
    log_figure(
        mlflow_client, run_id, metrics_prefix, column_name, silo_index, "density", fig
    )


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
    silo_index,
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

    with mlflow.start_run() as mlflow_run:
        # get Mlflow client and root run id
        mlflow_client = mlflow.tracking.client.MlflowClient()
        logger.debug(f"Root runId: {mlflow_run.data.tags.get('mlflow.rootRunId')}")
        root_run_id = mlflow_run.data.tags.get("mlflow.rootRunId")

        logger.debug(f"Loading data...")
        train_df = pd.read_csv(training_data)
        test_df = pd.read_csv(testing_data)

        log_metrics(
            mlflow_client,
            root_run_id,
            metrics_prefix,
            "summary/train",
            silo_index,
            "number_of_datapoints",
            f"{train_df.shape[0]}",
        )
        log_metrics(
            mlflow_client,
            root_run_id,
            metrics_prefix,
            "summary/test",
            silo_index,
            "number_of_datapoints",
            f"{test_df.shape[0]}",
        )

        # We are merging one hot encoded columns and creating categorical columns
        categorical_columns.extend([prefix[:-1] for prefix in onehot_columns_prefix])

        logger.debug(f"Train data analysis...")
        train_df = merge_onehot(train_df, onehot_columns_prefix)
        log_figure(
            mlflow_client,
            root_run_id,
            metrics_prefix,
            "summary/train",
            silo_index,
            "correlation",
            get_corr_fig(train_df),
        )

        for column in train_df.columns:
            logger.info(f"Processing training dataset column: {column}")
            if column in categorical_columns:
                analyze_categorical_column(
                    train_df,
                    column,
                    silo_index,
                    mlflow_client,
                    root_run_id,
                    f"{metrics_prefix}/train",
                )
            elif is_numeric_dtype(train_df[column]):
                analyze_numerical_column(
                    train_df,
                    column,
                    silo_index,
                    mlflow_client,
                    root_run_id,
                    f"{metrics_prefix}/train",
                )
            else:
                logger.info(f"Skipping analysis for column: {column}")

        logger.debug(f"Test data analysis...")
        test_df = merge_onehot(test_df, onehot_columns_prefix)
        log_figure(
            mlflow_client,
            root_run_id,
            metrics_prefix,
            "summary/test",
            silo_index,
            "correlation",
            get_corr_fig(test_df),
        )

        for column in test_df.columns:
            logger.info(f"Processing test dataset column: {column}")
            if column in categorical_columns:
                analyze_categorical_column(
                    test_df,
                    column,
                    silo_index,
                    mlflow_client,
                    root_run_id,
                    f"{metrics_prefix}/test",
                )
            elif is_numeric_dtype(test_df[column]):
                analyze_numerical_column(
                    test_df,
                    column,
                    silo_index,
                    mlflow_client,
                    root_run_id,
                    f"{metrics_prefix}/test",
                )
            else:
                logger.info(f"Skipping analysis for column: {column}")


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
            silo_index=args.silo_index,
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
