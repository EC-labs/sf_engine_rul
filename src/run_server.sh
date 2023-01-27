#! /bin/bash

set -e

script_directory="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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
	--name fedadapt_server \
  --shm-size 4G \
  fedadapt/base_image "script_rul_engine_server"
