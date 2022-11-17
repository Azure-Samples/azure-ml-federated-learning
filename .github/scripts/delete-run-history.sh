#!/bin/bash

job_name_with_created_date=$(az ml job list -g $1 -w $2 | jq ".[] | .name, .creation_context.created_at") 
echo $job_name_with_created_date
job_name=""
i=1 
for item in $job_name_with_created_date; do
    item=`sed -e 's/^"//' -e 's/"$//' <<< "$item"`
    if [[ $i%2 -eq 1 ]]; then
        job_name=$item
        i=$((i+1))
        continue
    fi    
    
    num_of_days=$((($(date +%s) - $(date -d $item +%s)) / (60 * 60 * 24) ))
    echo Job name: $job_name, Number of days: $num_of_days
    
    # Archive jobs that are older than 60 days
    if [[ $num_of_days -gt 60 ]]; then
        az ml job archive -g $1 -w $2 -n $job_name
    else
        echo "Number of days are less than 60."
    fi    
    i=$((i+1))
done


