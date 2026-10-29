#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

submit_job() {
    local stage="$1"
    local mode="$2"
    local size="$3"
    local dependency="${4:-}"
    local driver index job_name command

    if [[ "$mode" == "sparse" ]]; then
        driver="sparse/simulation_sparse.jl"
    else
        driver="simulations.jl"
    fi

    if [[ "$stage" == "667" ]]; then
        index=6
        job_name="m2_667_${mode}_${size}"
    else
        index=11
        job_name="m2_acc_${mode}_${size}"
    fi

    command="cd '$REPO_ROOT' && source ~/.bashrc && module load julia && module load gurobi && julia -t 8 '$driver' '$size' '$index'"

    local args=(
        --parsable
        --job-name="$job_name"
        --partition=mit_normal
        --cpus-per-task=8
        --output="$LOG_DIR/%x_%j.out"
        --error="$LOG_DIR/%x_%j.err"
    )

    if [[ "$stage" == "667" ]]; then
        args+=(--mem=16G)
        args+=(--time=00:05:00)
    else
        args+=(--mem=32G)
        args+=(--time=01:00:00)
        if [[ -n "$dependency" ]]; then
            args+=("--dependency=afterok:$dependency")
        fi
    fi

    sbatch "${args[@]}" --wrap="$command"
}

submit_size() {
    local mode="$1"
    local size="$2"
    local run_id acc_id

    run_id="$(submit_job 667 "$mode" "$size")"
    run_id="${run_id%%;*}"
    acc_id="$(submit_job accumulate "$mode" "$size" "$run_id")"
    acc_id="${acc_id%%;*}"

    printf '%s size %s: efficient_667 job %s -> accumulate job %s\n' \
        "$mode" "$size" "$run_id" "$acc_id"
}

for size in 10 100 1000 10000 100000; do
    submit_size sparse "$size"
done

for size in 10 30 100 300 1000; do
    submit_size dense "$size"
done