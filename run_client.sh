#! /bin/bash

set -e

docker build \
  -t fedadapt/base_image .

docker run \
  -it --name fedadapt_client --rm \
  -h pi41 \
  --network fedadapt_network \
  fedadapt/base_image "fl_training.fedadapt_client_run.py"
