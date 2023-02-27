#! /bin/bash

script_directory="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker build \
  -t fedadapt/base_image "${script_directory}"

docker run \
  -it --name test_rul --rm \
  -v "${script_directory}/../data:/usr/src/app/data" \
  -v "${script_directory}/logs:/usr/src/app/logs" \
  -v "${script_directory}/../models:/usr/src/app/trained" \
  fedadapt/base_image pdb -m models.turbofan # "script_rul.py"
