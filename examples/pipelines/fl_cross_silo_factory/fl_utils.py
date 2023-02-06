from azure.ai.ml import Output, Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.constants import AssetTypes

def custom_fl_data_path(
        datastore_name, output_name, unique_id="${{name}}", iteration_num=None
):
    """Produces a path to store the data during FL training.
    Args:
        datastore_name (str): name of the Azure ML datastore
        output_name (str): a name unique to this output
        unique_id (str): a unique id for the run (default: inject run id with ${{name}})
        iteration_num (str): an iteration number if relevant
    Returns:
        data_path (str): direct url to the data path to store the data
    """
    data_path = f"azureml://datastores/{datastore_name}/paths/federated_learning/{output_name}/{unique_id}/"
    if iteration_num is not None:
        data_path += f"iteration_{iteration_num}/"

    return data_path


def scatter_gather_iteration(
        scatter,
        gather,
        scatter_strategy,
        gather_strategy,
        scatter_to_gather_map=None,
        gather_to_scatter_map=None,
        accumulators=None,
        iterations=None,
        scatter_constant_inputs=None,
):
    @pipeline(name="Scatter Gather Iteration")
    def scatter_gather_iter(**kwargs):
        iteration_num = kwargs.pop("iteration_num")
        gather_inputs = {}
        for silo_index in range(1, len(scatter_strategy) + 1):
            silo_config = scatter_strategy[silo_index - 1]

            scatter_input = {
                **silo_config["inputs"], **scatter_constant_inputs,
                **{"scatter_compute": silo_config["compute"]}
            }

            scatter_gather = scatter(**scatter_input, iteration_num=iteration_num, **kwargs)
            # scatter_gather = scatter(**scatter_input, checkpoint=checkpoint)

            # TODO: Assumption that scatter subgraph produces output with name "model"
            scatter_gather.outputs.model = Output(
                type=AssetTypes.URI_FOLDER,
                mode="mount",
                path=custom_fl_data_path(
                    gather_strategy["datastore"],
                    f"model/silo{silo_index}",
                    iteration_num=iteration_num.result()  # change this once iteration loop is added
                ),
            )

            # TODO: Hack for POC should not be used for prod code
            # Check with pipeline team on support for overriding compute and datastore for subgraphs
            # Until then compute need to be explicitly passed to child components as a param
            # This adds requirement that components used in FL should accept compute as param and
            # handle it appropriately
            for job_name, job in scatter_gather.component.jobs.items():
                job.compute = silo_config["compute"]

            # TODO: scatter_to_gather_map is user provided and no good way to know what inputs needed
            # What is the expectation for it ?
            for scatter_output_name, gather_input_name in scatter_to_gather_map.items():
                gather_inputs[gather_input_name(scatter_output_name, silo_index)] = scatter_gather.outputs[scatter_output_name]
            # gather_inputs[scatter_to_gather_map("model", silo_index)] = scatter_gather.outputs.model

        gather_instance = gather(**gather_inputs)

        gather_instance.outputs.aggregated_output = Output(
            type=AssetTypes.URI_FOLDER,
            mode="mount",
            path=custom_fl_data_path(
                gather_strategy["datastore"],
                f"model_aggregated",
                iteration_num=iteration_num.result()
            ),
        )

        # TODO: Hack for POC should not be used for prod code
        # Check with pipeline team on support for overriding compute and datastore for subgraphs
        # Until then compute need to be explicity passed to child components as a param
        # This adds requirement that components used in FL should accept compute as param and
        # handle it appropriately
        gather_instance.compute = gather_strategy["compute"]

        # return {
        #     # Assuming gather should return aggregated_model as output(Need to change in this component)
        #     "model": gather_instance.outputs.aggregated_output,
        # }

        return gather_instance.outputs

    @pipeline(name="FL Pipeline")
    def fl_pipeline():
        checkpoint = None
        scatter_inputs = {}

        def get_scatter_inputs(gather_to_scatter_map, gather_iteration_body):
            scatter_inputs = {}
            for gather_output_name, scatter_input_name in gather_to_scatter_map.items():
                scatter_inputs[scatter_input_name] = gather_iteration_body.outputs[gather_output_name]
            return scatter_inputs

        for i in range(0, iterations):
            iteration_body = scatter_gather_iter(iteration_num=i, **scatter_inputs)
            iteration_body.outputs.aggregated_output = Output(
                type=AssetTypes.URI_FOLDER,
                mode="mount",
                path=custom_fl_data_path(
                    gather_strategy["datastore"],
                    f"model_aggregated",
                    iteration_num=i
                ),
            )

            # Check output type

            scatter_inputs = get_scatter_inputs(gather_to_scatter_map, iteration_body)

            # checkpoint = iteration_body.outputs.model

        # return {
        #     "final_model": iteration_body.outputs.model,
        # }
        return iteration_body.outputs

    fl_pipeline_instance = fl_pipeline()

    # set this based on output type as uri_folder
    # fl_pipeline_instance.outputs.final_model = Output(
    #     type=AssetTypes.URI_FOLDER,
    #     mode="mount",
    #     path=custom_fl_data_path(
    #         gather_strategy["datastore"],
    #         f"final_model",
    #     ),
    # )

    return fl_pipeline_instance