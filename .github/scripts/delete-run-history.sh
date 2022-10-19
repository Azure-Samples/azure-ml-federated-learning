#!/bin/bash

outputs=$(az ml job list -g fl-pipeline-testing-rg -w aml-flpipelinetest | jq "map(.name, .creation_context.created_at)") 
echo $outputs
echo "Done"
