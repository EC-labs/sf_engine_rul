#! /bin/bash

set -e

script_directory="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

program_name="rul_engine"

docker build \
  -t fedadapt/base_image "$script_directory"

docker run \
  -it \
  --label "group=sf_engine_rul" \
  --rm \
  --env "NCLIENTS=1" \
  --network fedadapt_network \
  -v "${script_directory}/../results:/usr/src/app/results" \
  -v "${script_directory}/../data:/usr/src/app/data" \
  -v "${script_directory}/logs:/usr/src/app/logs" \
  --env "ENGINE=20" \
  --env "PROGRAM_NAME=${program_name}" \
  --name fedadapt_client \
  --shm-size 4G \
  fedadapt/base_image "script_${program_name}_client"
