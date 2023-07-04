SHELL = /bin/bash

# Modify these variables
CPUS="1"
NCLIENTS=1
PROGRAM=rul_engine
CENTRALIZED_PROGRAM=rul_turbofan
ISOLATED_ENGINE=2
TEST_PROGRAM_DIRECTORY=
FAULTY=0
FAULTY_CLIENT=[]
NOISE_AMPLITUDE=10


# Do not modify these variables
ROOTDIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
SRCDIR=$(ROOTDIR)/src

SCRIPT=script_$(PROGRAM)
GROUP_LABEL=group=sf_engine_rul
NETWORK="fedadapt_network"
IMAGE="fedadapt/base_image"
EXEC_TIME=$(shell date +'%Y-%m-%d_%H:%M:%S.%s')

CONTAINER_LABELS=--label "$(GROUP_LABEL)"
CONTAINER_NETWORK=--network $(NETWORK)
COMMON_ENVIRONMENT=--env "NCLIENTS=$(NCLIENTS)" --env "NOISE_AMPLITUDE=${NOISE_AMPLITUDE}"
VOLUME_DATA=-v "$(ROOTDIR)/data:/usr/src/app/data"
VOLUME_RESULTS=-v "$(ROOTDIR)/results:/usr/src/app/results"
VOLUME_LOGS=-v "$(SRCDIR)/logs:/usr/src/app/logs"
CPUS_FLAG=--cpus=$(CPUS)
COMMON_FLAGS=--rm --shm-size 4G
BASE_LOGS=$(SRCDIR)/logs
LOGS_DIR=$(BASE_LOGS)/$(EXEC_TIME)

.PHONY: clean_resources clean_logs clean \
				create_network create_image run

all: run

run: create_image create_network
		@mkdir -p "$(LOGS_DIR)"
		docker run \
			$(CONTAINER_LABELS) \
			$(COMMON_FLAGS) \
			$(CPUS_FLAG) \
			$(COMMON_ENVIRONMENT) \
			$(CONTAINER_NETWORK) \
			$(VOLUME_RESULTS) $(VOLUME_DATA) $(VOLUME_LOGS) \
			--env PROGRAM_NAME=$(PROGRAM) \
			--env FAULTY=$(FAULTY) \
			--env FAULTY_CLIENT=$(FAULTY_CLIENT) \
			--name fedadapt_server \
			$(IMAGE) $(SCRIPT)_server 1>"$(LOGS_DIR)/server.log" 2>&1 &
		@sleep 3
		ENGINES=(2 5 10 16 18 20); \
		for i in $$(seq 0 $$(($(NCLIENTS) - 1))); do \
			docker run \
				$(CONTAINER_LABELS) \
				$(COMMON_FLAGS) \
				$(CPUS_FLAG) \
				$(COMMON_ENVIRONMENT) \
				$(CONTAINER_NETWORK) \
				$(VOLUME_RESULTS) $(VOLUME_DATA) $(VOLUME_LOGS) \
				--env PROGRAM_NAME=$(PROGRAM) \
				--env ENGINE=$${ENGINES[$$i]} \
				--env FAULTY=$(FAULTY) \
				--env FAULTY_CLIENT=$(FAULTY_CLIENT) \
				--name fedadapt_client_$$i \
				$(IMAGE) $(SCRIPT)_client 1>"$(LOGS_DIR)/client_$$i.log" 2>&1 & \
			sleep 0.5; \
		done

create_image: 
		@docker build -t $(IMAGE) $(SRCDIR)

create_network: 
		@[[ -z "$$( docker network ls --format "{{.Name}}" | awk '{ if($$1 == $(NETWORK)) { print $$1 } }' )"  ]] \
			 && docker network create -d bridge $(NETWORK) || true

run_centralized: create_image
		docker run \
			$(CONTAINER_LABELS) \
			$(COMMON_FLAGS) \
			$(CPUS_FLAG) \
			$(VOLUME_RESULTS) $(VOLUME_DATA) $(VOLUME_LOGS) \
			--name turbofan_centralized \
			--env ENGINE=$(ISOLATED_ENGINE) \
			--env PROGRAM_NAME=$(CENTRALIZED_PROGRAM) \
			-it \
			$(IMAGE) script_$(CENTRALIZED_PROGRAM)
	
test_model: create_image
		docker run \
			$(CONTAINER_LABELS) \
			$(COMMON_FLAGS) \
			$(CPUS_FLAG) \
			$(VOLUME_RESULTS) $(VOLUME_DATA) $(VOLUME_LOGS) \
			--name test_model \
			--env PROGRAM_NAME=test_model \
			$(IMAGE) script_test_model $(TEST_PROGRAM_DIRECTORY)

clean_resources:
		cnts=($$(docker ps -a --filter 'label=$(GROUP_LABEL)' | awk '{if(NR > 1) { print $$1 } }')); \
		(( $${#cnts[@]} > 0 )) \
		&& docker stop "$${cnts[@]}" || true
		@[[ -z "$$( docker network ls --format "{{.Name}}" | awk '{ if($$1 == $(NETWORK)) { print $$1 } }' )"  ]] \
		   || { echo "Removing network"; docker network rm $(NETWORK); }
	
clean_logs:
		[[ -d "$(BASE_LOGS)" ]] && rm -r "$(BASE_LOGS)" || true

clean: clean_resources clean_logs
