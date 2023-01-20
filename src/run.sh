#! /bin/bash

# Fail if any of the following commands terminates w/ non-zero exit code
set -e 

execTime="$(date +'%Y-%m-%d_%H:%M:%S.%s')"
logsDir="logs/${execTime}"
mkdir -p "$logsDir"

existsNet="$(docker network ls | awk '{if(NR > 1) {if($2 == "fedadapt_network") print $2}}')"

if [[ -z existsNet ]]; then
  docker network create -d bridge fedadapt_network
fi

docker build \
  -t fedadapt/base_image .

read -p "How many clients? " nClients
if ! [[ "$nClients" =~ [0-9]+ ]]; then
  echo "Number of clients should be an integer"; exit 1;
fi

echo "Start Server" 
docker run \
  --name fedadapt_server --rm \
  --network fedadapt_network \
  --env "NCLIENTS=${nClients}" \
  fedadapt/base_image "fl_training.fedadapt_server_run.py" 1>"${logsDir}/server.log" 2>&1 &

sleep 3

for i in $( seq 0 $((nClients - 1)) ); do
  echo "Start Client $i"
  docker run \
    --name "fedadapt_client_${i}" --rm \
    -h pi41 \
    --network fedadapt_network \
    --env "ENGINE=${i}" \
    --env "NCLIENTS=${nClients}" \
    fedadapt/base_image "fl_training.fedadapt_client_run.py" 1>"${logsDir}/client_${i}.log" 2>&1 &
  done

