set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# === Parameters ===
if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_folder> [docker_image_tag]"
    exit 1
fi

INPUT_DIR="$1"
DOCKER_IMAGE_TAG="${2:-unicorn_baseline}"   # Use $2 if given, else default

DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

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