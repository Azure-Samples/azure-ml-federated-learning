import os
import shutil
from mldesigner import command_component
from mldesigner import Input
from mldesigner import Output

@command_component()
def scatter_fusion(
    fusion_outputs: Output(type="uri_folder"),
    **silo_outputs
):
    os.makedirs(fusion_outputs, exist_ok=True)

    for k, v in silo_outputs.items():
        shutil.copytree(v, os.path.join(fusion_outputs, k))
