#!/bin/bash

# ------------------------------------------------------------------------------
# Bash script to automate running local test for all UNICORN tasks using few-shot data from Zenodo.
#
# This script loops through a list of ZIP files (one per task), downloads data 
# if necessary, and runs the test routine for each task using `run_task_local.sh`.
#
# To add tasks, simply uncomment or add ZIP filenames to the TASK_ZIPS array.
#
# Example usage:
#    ./run_all_local.sh
# -------

# Base URL for all tasks
BASE_URL="https://zenodo.org/record/15112095/files"

# List of all ZIP filenames (just copy the names as they appear)
TASK_ZIPS=(
    Task01_classifying_he_prostate_biopsies_into_isup_scores.zip
    # Task02_classifying_lung_nodule_malignancy_in_ct.zip
    #Task03_predicting_the_time_to_biochemical_recurrence_in_he_prostatectomies.zip
    # Task04_predicting_slide_level_tumor_proportion_score_in_ihc_stained_wsi.zip
     Task05_detecting_signet_ring_cells_in_he_stained_wsi_of_gastric_cancer.zip
    # Task06_detecting_clinically_significant_prostate_cancer_in_mri_exams.zip
    # Task07_detecting_lung_nodules_in_thoracic_ct.zip
    # Task08_detecting_mitotic_figures_in_breast_cancer_wsis.zip
    # Task09_segmenting_rois_in_breast_cancer_wsis.zip
    # Task10_segmenting_lesions_within_vois_in_ct.zip
    # Task11_segmenting_three_anatomical_structures_in_lumbar_spine_mri.zip
    # Task12_predicting_histopathology_sample_origin.zip
    # Task13_classifying_pulmonary_nodule_presence.zip
    # Task14_classifying_kidney_abnormality.zip
    # Task15_hip_kellgren_lawrence_score.zip
    # Task16_classifying_colon_histopathology_diagnosis.zip
    # Task17_predicting_lesion_size_measurements.zip
    # Task18_predicting_prostate_volume_psa_and_psa_density.zip
    # Task19_anonymizing_report.zip
    # Task20_generating_caption_from_wsi.zip
)

for ZIP in "${TASK_ZIPS[@]}"; do
    echo "Running local test for task: $ZIP"
    ./run_task_local.sh "${BASE_URL}/${ZIP}"
    echo
done

