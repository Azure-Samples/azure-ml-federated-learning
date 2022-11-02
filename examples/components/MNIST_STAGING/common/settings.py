# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
from distutils.util import strtobool

from azureml.automl.dnn.vision.common import utils
from azureml.automl.dnn.vision.common.constants import SettingsLiterals


class CommonSettings:

    def __init__(self, training_data: str, validation_data: str, model_output: str) -> None:
        self.training_data = training_data
        self.validation_data = validation_data
        self.model_output = model_output

    @classmethod
    def create_from_parsing_current_cmd_line_args(cls) -> "CommonSettings":
        parser = argparse.ArgumentParser()
        parser.add_argument(utils._make_arg('training_data'), type=str)
        parser.add_argument(utils._make_arg('validation_data'), type=str)
        parser.add_argument(utils._make_arg('model_output'), type=str)
        args, _ = parser.parse_known_args()
        return CommonSettings(
            args.training_data, args.validation_data, args.model_output
        )


class ClassificationSettings(CommonSettings):

    def __init__(self, training_data: str, validation_data: str, model_output: str, multilabel: bool) -> None:
        super().__init__(training_data, validation_data, model_output)
        self.multilabel = multilabel

    @classmethod
    def create_from_parsing_current_cmd_line_args(cls) -> "ClassificationSettings":
        # Create common settings
        common_settings = CommonSettings.create_from_parsing_current_cmd_line_args()

        # Create classification settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--multilabel', type=lambda x: bool(strtobool(str(x))))
        args, _ = parser.parse_known_args()
        return ClassificationSettings(
            common_settings.training_data, common_settings.validation_data, common_settings.model_output,
            args.multilabel)

class CheckpointSettings(CommonSettings):

    @classmethod
    def create_from_parsing_current_cmd_line_args(cls) -> "ClassificationSettings":
        # Create common settings
        common_settings = CommonSettings.create_from_parsing_current_cmd_line_args()

        # Create classification settings
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_filename', type=str, required=False)
        parser.add_argument('--checkpoint_dataset_id', type=str, required=False)
        parser.add_argument('--checkpoint_run_id', type=str, required=False)
        args, _ = parser.parse_known_args()

        checkpoint_setting = {}
        if args.checkpoint_filename:
            checkpoint_setting[SettingsLiterals.CHECKPOINT_FILENAME] = args.checkpoint_filename 
        if args.checkpoint_dataset_id:
            checkpoint_setting[SettingsLiterals.CHECKPOINT_DATASET_ID] = args.checkpoint_dataset_id 
        if args.checkpoint_run_id:
            checkpoint_setting[SettingsLiterals.CHECKPOINT_RUN_ID] = args.checkpoint_run_id 
        
        return checkpoint_setting


class ObjectDetectionSettings(CommonSettings):
    pass


class InstanceSegmentationSettings(CommonSettings):
    pass
