#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

submit_job() {
    local stage="$1"
    local index="$2"
    local job_name="$3"
    local memory="$4"
    local dependency="${5:-}"
    local command

    command="cd '$REPO_ROOT' && source ~/.bashrc && module load julia && module load gurobi && julia -t 8 'sparse/simulation_sparse.jl' '100000' '$index'"

    local args=(
        --parsable
        --job-name="$job_name"
        --partition=mit_normal
        --cpus-per-task=8
        --mem="$memory"
        --output="$LOG_DIR/%x_%j.out"
        --error="$LOG_DIR/%x_%j.err"
    )

    if [[ "$stage" == "667" ]]; then
        args+=(--time=00:05:00)
    else
        args+=(--time=01:00:00)
        args+=("--dependency=afterok:$dependency")
    fi

    sbatch "${args[@]}" --wrap="$command"
}

run_id="$(submit_job 667 6 m2_667_sparse_100000 32G)"
run_id="${run_id%%;*}"
acc_id="$(submit_job accumulate 11 m2_acc_sparse_100000 32G "$run_id")"
acc_id="${acc_id%%;*}"

printf 'sparse size 100000: efficient_667 job %s -> accumulate job %s\n' \
    "$run_id" "$acc_id"
