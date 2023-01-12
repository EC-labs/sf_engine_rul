#! /bin/bash

set -e

docker build \
  -t fedadapt/base_image .

docker run \
  -it --name fedadapt_server --rm \
  --network fedadapt_network \
  --env "NCLIENTS=1" \
  fedadapt/base_image "fl_training.fedadapt_server_run.py"
