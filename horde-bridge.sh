#!/bin/bash
# Get the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Prevent NVIDIA driver file-cache init errors ("driverInitFileInfo ... result=11")
export CUDA_CACHE_DISABLE="${CUDA_CACHE_DISABLE:-1}"
# Defer CUDA module loading to reduce startup errors with mismatched driver versions
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"

# Build the absolute path to the Conda environment
CONDA_ENV_PATH="$SCRIPT_DIR/conda/envs/linux/lib"

# Add the Conda environment to LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_ENV_PATH:$LD_LIBRARY_PATH"

# List of directories to check
dirs=(
    "/usr/lib"
    "/usr/local/lib"
    "/lib"
    "/lib64"
    "/usr/lib/x86_64-linux-gnu"
)

# Check each directory
for dir in "${dirs[@]}"; do
    if [ -f "$dir/libjemalloc.so.2" ]; then
        export LD_PRELOAD="$dir/libjemalloc.so.2"
        printf "Using jemalloc from $dir\n"
        break
    fi
done

# If jemalloc was not found, print a warning
if [ -z "$LD_PRELOAD" ]; then
    printf "WARNING: jemalloc not found. You may run into memory issues! We recommend running 'sudo apt install libjemalloc2'\n"
    # Press q to quit or any other key to continue
    read -n 1 -s -r -p "Press q to quit or any other key to continue: " key
    if [ "$key" = "q" ]; then
        printf "\n"
        exit 1
    fi
fi

if "$SCRIPT_DIR/runtime.sh" python -s "$SCRIPT_DIR/download_models.py"; then
    echo "Model Download OK. Starting worker..."
    while true; do
        "$SCRIPT_DIR/runtime.sh" python -s "$SCRIPT_DIR/run_worker.py" $*
        exit_code=$?
        # Exit code 42 (consts.WORKER_RESTART_EXIT_CODE) means the worker requested a restart.
        # os.execv() normally restarts in-place without returning here, but if the shutdown
        # watchdog had to force-kill the process instead, control returns to this script and
        # the worker must be re-launched explicitly.
        if [ "$exit_code" -eq 42 ]; then
            echo "Worker requested a restart. Restarting..."
            continue
        fi
        break
    done
else
    echo "download_models.py exited with error code. Aborting"
fi
