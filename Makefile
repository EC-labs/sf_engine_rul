SHELL = /bin/bash

NCLIENTS=1
GROUP_LABEL=group=sf_engine_rul
NETWORK="fedadapt_network"
IMAGE="fedadapt/base_image"
EXEC_TIME=$(shell date +'%Y-%m-%d_%H:%M:%S.%s')

CONTAINER_LABELS=--label "$(GROUP_LABEL)"
CONTAINER_NETWORK=--network $(NETWORK)
COMMON_ENVIRONMENT=--env "NCLIENTS=$(NCLIENTS)"
COMMON_FLAGS=--rm
LOGS_DIR=logs/$(EXEC_TIME)


.PHONY: clean_resources clean_logs clean \
				create_network create_image run

run: create_image create_network
		@mkdir -p "$(LOGS_DIR)"
		docker run \
			$(CONTAINER_LABELS) \
			$(COMMON_FLAGS) \
			$(COMMON_ENVIRONMENT) \
			$(CONTAINER_NETWORK) \
			--name fedadapt_server \
			$(IMAGE) fl_training.fedadapt_server_run 1>"$(LOGS_DIR)/server.log" 2>&1 &
		@sleep 3
		for i in $$(seq 0 $$(($(NCLIENTS) - 1))); do \
			docker run \
				$(CONTAINER_LABELS) \
				$(COMMON_FLAGS) \
				$(COMMON_ENVIRONMENT) \
				$(CONTAINER_NETWORK) \
				--env ENGINE=$$i \
				--name fedadapt_client_$$i \
				$(IMAGE) fl_training.fedadapt_client_run 1>"$(LOGS_DIR)/client_$$i.log" 2>&1 & \
		done

create_image: 
		@docker build -t $(IMAGE) . 1>/dev/null 2>&1

create_network: 
		@[[ -z "$$( docker network ls --format "{{.Name}}" | awk '{ if($$1 == $(NETWORK)) { print $$1 } }' )"  ]] \
			 && docker network create -d bridge $(NETWORK) || true

clean_resources:
		@cnts=($$(docker ps -a --filter 'label=$(GROUP_LABEL)' | awk '{if(NR > 1) { print $$1 } }')); \
 		 for cnt in $${cnts[@]}; do \
			 echo "Removing container $$cnt"; \
			 docker stop "$$cnt" 1>/dev/null || true; \
			 docker rm "$$cnt" || true; \
		 done
		@[[ -z "$$( docker network ls --format "{{.Name}}" | awk '{ if($$1 == $(NETWORK)) { print $$1 } }' )"  ]] \
		   || { echo "Removing network "; docker network rm $(NETWORK); }
	
clean_logs:
		@rm -r logs/

clean: clean_resources clean_logs
