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
  -v "$script_directory/../results:/usr/src/app/results" \
  -v "$script_directory/../data:/usr/src/app/data" \
	--name fedadapt_server \
  fedadapt/base_image pdb -m "distributed_learning.fedadapt_server_run"
