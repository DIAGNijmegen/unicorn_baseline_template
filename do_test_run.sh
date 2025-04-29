#!/usr/bin/env bash

# ============================================================================
# do_test_run.sh
#
# This script performs a test run of a Docker container.
#
# Usage:
#    ./do_test_run.sh <TASK_FOLDER_OR_ZIP> [DOCKER_IMAGE_TAG]
#
# Description:
# 1. **Local Development with Public Shots**:
#    - Provide the task folder or zip file containing the public shots data as the first argument.
#    - Optionally, provide the Docker image tag as the second argument.
#    - Example:
#      ./do_test_run.sh /path/to/task_folder unicorn_template
#
# 2. **On the Platform (e.g. GC)**:
#    - You may run the script with just the input task folder:
#      ./do_test_run.sh /path/to/task_folder
#
# Arguments:
#    $1 - Task folder or zip file (required)
#    $2 - Docker image tag (optional, default: unicorn_template)
#
# ============================================================================

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# === Argument parsing ===
if [ $# -lt 1 ]; then
    echo "Error: Missing required argument <TASK_FOLDER_OR_ZIP>"
    echo "Usage: $0 <TASK_FOLDER_OR_ZIP> [DOCKER_IMAGE_TAG]"
    exit 1
fi

INPUT_DIR="$1"
DOCKER_IMAGE_TAG="${2:-unicorn_template}"

echo "Using INPUT_DIR: $INPUT_DIR"
echo "Using DOCKER_IMAGE_TAG: $DOCKER_IMAGE_TAG"

OUTPUT_DIR="${SCRIPT_DIR}/test/output"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

echo "=+= (Re)build the container"

source "${SCRIPT_DIR}/do_build.sh" "$DOCKER_IMAGE_TAG"

cleanup() {
    echo "=+= Cleaning permissions ..."
    docker run --rm \
      --platform=linux/amd64 \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "chmod -R -f o+rwX /output/* || true"
}

if [ -d "$OUTPUT_DIR" ]; then
  chmod -f o+rwx "$OUTPUT_DIR"
  echo "=+= Cleaning up any earlier output"
  docker run --rm \
      --platform=linux/amd64 \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      "$DOCKER_IMAGE_TAG" \
      -c "rm -rf /output/* || true"
else
  mkdir -m o+rwx "$OUTPUT_DIR"
fi

trap cleanup EXIT

echo "=+= Doing a forward pass"
docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null
docker run --rm \
    --platform=linux/amd64 \
    --network none \
    --gpus all \
    --volume "$INPUT_DIR":/input:ro \
    --volume "$OUTPUT_DIR":/output \
    --volume "$DOCKER_NOOP_VOLUME":/tmp \
    --volume "${SCRIPT_DIR}/model":/opt/ml/model/:ro \
    "$DOCKER_IMAGE_TAG"
docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null

echo "=+= Wrote results to ${OUTPUT_DIR}"
echo "=+= Save this image for uploading via ./do_save.sh \"${DOCKER_IMAGE_TAG}\""