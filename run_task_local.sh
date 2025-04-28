#!/bin/bash

# ------------------------------------------------------------------------------
# Bash script to simplify local development using public few-shot data 
# for a single task from Zenodo.
#
# Steps performed by this script:
#   1. Checks if the task data is already unzipped locally.
#   2. If not, checks if the zip file has already been downloaded.
#   3. If not, downloads the zip file from Zenodo and unzips it to the local data directory.
#   4. Performs a local test run (`./do_test_run.sh`) for an example shot.
#
# Example for Task 1:
#   BASE_URL="https://zenodo.org/record/15112095/files"
#   ZIP="Task01_classifying_he_prostate_biopsies_into_isup_scores.zip"
#
# Usage:
#   ./run_task_local.sh "${BASE_URL}/${ZIP}"
# ------------------------------------------------------------------------------

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <ZIP_FILE_URL>"
    exit 1
fi

FILE_URL="$1"
OUTPUT_NAME=$(basename "${FILE_URL%%\?*}")  
TASK_NAME="${OUTPUT_NAME%.zip}"
LOCAL_DATA_DIR="local_data"
TASK_FOLDER="${LOCAL_DATA_DIR}/${TASK_NAME}"

# Ensure the local_data directory exists
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    mkdir "$LOCAL_DATA_DIR"
    echo "Created directory: $LOCAL_DATA_DIR"
fi

# 1. Check if already unzipped
if [ -d "${TASK_FOLDER}/shots-public" ]; then
    echo "Data already extracted at ${TASK_FOLDER}/shots-public, nothing to do."
else
    # 2. If not unzipped, check for zip file
    if [ -f "$OUTPUT_NAME" ]; then
        echo "ZIP file $OUTPUT_NAME exists, extracting to $TASK_FOLDER ..."
    else
        # 3. If zip file is missing, download it then unzip
        echo "Downloading $OUTPUT_NAME from $FILE_URL ..."
        curl -L -o "$OUTPUT_NAME" "$FILE_URL"
    fi
    mkdir -p "$TASK_FOLDER"
    unzip -o "$OUTPUT_NAME" -d "$TASK_FOLDER"
    echo "Unzipped to $TASK_FOLDER"
fi

# NOTE: The following rename is ONLY for Task01_classifying_he_prostate_biopsies_into_isup_scores.
# This is due to a known bug with the directory name ("prostate-tissue-biopsy-wsi" should be
# "prostate-tissue-biospy-wsi" to be consistent with GC platform).
if [ "$TASK_NAME" == "Task01_classifying_he_prostate_biopsies_into_isup_scores" ]; then
    find "${TASK_FOLDER}/shots-public" -type d -name "prostate-tissue-biopsy-wsi" | while read dir; do
        new_dir="${dir%prostate-tissue-biopsy-wsi}prostate-tissue-biospy-wsi"
        if [ ! -d "$new_dir" ]; then
            echo "Renaming images/prostate-tissue-biopsy-wsi to images/prostate-tissue-biospy-wsi in all cases..."
            mv "$dir" "$new_dir"
            echo "Renamed $dir -> $new_dir"
        else
            echo "Skip $dir, already exists as $new_dir"
        fi
    done
fi


# Find the first case_id in shots-public
SHOTS_PUBLIC="${TASK_FOLDER}/shots-public"
FIRST_CASE=$(ls "$SHOTS_PUBLIC" | head -n 1)
echo "Processing case: $FIRST_CASE"


# Mount the CONTENTS of the case to test/input/
ABS_PATH=$(realpath "./${SHOTS_PUBLIC}/${FIRST_CASE}")

# Perform the test run with a public shot
./do_test_run.sh "$ABS_PATH"