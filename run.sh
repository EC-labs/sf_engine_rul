#! /bin/bash

frequency=20
faulty_client='[5]'
evaluation_directory="evaluation=2023-07-04"
noise_amplitude=1


check_server_status () {
    docker container inspect -f '{{.State.Status}}' "$1" 1>/dev/null 2>&1
}

wait_server_stop () {
    if ! check_server_status "$1"; then
        return 0
    fi 

    echo "Waiting for $1"
    while check_server_status "$1"; do
        for i in {0..20}; do
            printf "."
            sleep 1
            if check_server_status "$1"; then 
                continue
            else
                break
            fi 
        done
        printf "\n"
    done
}

run_centralized () {
    echo "Run centralized turbofan"
    wait_server_stop "run_centralized"
    make run_centralized \
        CPUS=2 \
        CENTRALIZED_PROGRAM="rul_turbofan"
    wait_server_stop "run_centralized"
}

run_centralized_isolated () {
    echo "Run centralized isolated"
    wait_server_stop "run_centralized"
    temp_engines=(2 5 10 16 18 20)
    for engine in ${temp_engines}; do
        make run_centralized \
            CPUS=2 \
            CENTRALIZED_PROGRAM="rul_turbofan_isolated" \
            ISOLATED_ENGINE=${engine}
    done
    wait_server_stop "run_centralized"
}

run_distributed_scripts () {
    echo "Run distributed scripts"
    for script in "$@"; do
        wait_server_stop "fedadapt_server"
        make run \
            CPUS=2 \
            PROGRAM="$script" \
            NCLIENTS=6 
        echo "${script}"
        sleep 10
    done
    wait_server_stop "fedadapt_server"
}

run_faulty_distributed_scripts () {
    echo "Run faulty distributed scripts"
    for script in "$@"; do
        wait_server_stop "fedadapt_server"
        make run \
            CPUS=2 \
            PROGRAM="$script" \
            NCLIENTS=6 \
            FAULTY=1 \
            FAULTY_CLIENT="${faulty_client}"
            NOISE_AMPLITUDE="${noise_amplitude}"
        echo "${script}"
        sleep 10
    done
    wait_server_stop "fedadapt_server"
}

test_models () { 
    for test_program_directory in "$@"; do
        wait_server_stop "test_model"
        make test_model \
            CPUS=4 \
            TEST_PROGRAM_DIRECTORY="${test_program_directory}"
        echo $test_program_directory
        sleep 10
    done
    wait_server_stop "test_model"
}

present_results () {
    for program in "$@"; do
        echo "${program}"
        cat "results/${evaluation_directory}/${program}/test_metrics.json" |
            jq
    done
}

add_centralized_isolated () {
    local -n temp_engines="$1"
    local -n temp_models="$2"
    for engine in "${temp_engines[@]}"; do
        temp_models+=("frequency=${frequency}/program=rul_turbofan_isolated/engine=${engine}")
    done
}

add_distributed () {
    local -n temp_distributed_programs="$1"
    local -n temp_models="$2"
    for program in "${distributed_programs[@]}"; do
        temp_models+=("frequency=${frequency}/program=${program}")
    done

}

add_distributed_faulty () {
    local -n temp_distributed_programs="$1"
    local -n temp_models="$2"
    for program in "${distributed_programs[@]}"; do
        temp_models+=("frequency=${frequency}/faulty_client=${faulty_client}/program=${program}")
    done

}

distributed_programs=(
    "random_best"    
    "random_softmax"
    "full_best"
    "full_softmax"
    "rul_engine"    
)
engines=(2 5 10 16 18 20)

models_directory=(
    "frequency=${frequency}/program=rul_turbofan"
)
add_centralized_isolated  engines  models_directory
add_distributed  distributed_programs  models_directory
add_distributed_faulty  distributed_programs  models_directory
printf "%s\n" "${models_directory[@]}"

run_centralized 
run_centralized_isolated
run_distributed_scripts "${distributed_programs[@]}"
run_faulty_distributed_scripts "${distributed_programs[@]}"

test_models "${models_directory[@]}"
present_results "${models_directory[@]}"
