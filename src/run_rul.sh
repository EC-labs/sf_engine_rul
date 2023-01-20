#! /bin/bash

docker build \
  -t fedadapt/base_image .

docker run \
  -it --name test_rul --rm \
  -v "$(pwd)/../data:/usr/src/app/data" \
  -v "$(pwd)/logs:/usr/src/app/logs" \
  -v "$(pwd)/../models:/usr/src/app/trained" \
  fedadapt/base_image "pdb" "-m" "models.rul_cnn"
