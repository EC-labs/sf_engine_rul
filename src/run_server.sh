#! /bin/bash

set -e

docker build \
  -t fedadapt/base_image .

docker run \
  -it --name fedadapt_server --rm \
  --network fedadapt_network \
  --env "NCLIENTS=1" \
  fedadapt/base_image "distributed_learning.fedadapt_server_run.py"
