#! /bin/bash

script_directory="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker build \
  -t fedadapt/base_image "${script_directory}"

docker run \
  -it --rm --shm-size 4G \
  -v "${script_directory}/../data:/usr/src/app/data" \
  -v "${script_directory}/logs:/usr/src/app/logs" \
  -v "${script_directory}/../results:/usr/src/app/results" \
  -v "${script_directory}/../models:/usr/src/app/trained" \
  fedadapt/base_image script_rul_turbofan 
