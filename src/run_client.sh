#! /bin/bash

set -e

docker build \
  -t fedadapt/base_image .

docker run \
  -it --name fedadapt_client --rm \
  -h pi41 \
  --network fedadapt_network \
  --env "ENGINE=0" \
  fedadapt/base_image "distributed_learning.fedadapt_client_run.py"
