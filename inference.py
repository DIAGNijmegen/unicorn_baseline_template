#  Copyright 2025 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

import hashlib
import json
import warnings
from glob import glob
from pathlib import Path
from typing import Any, Iterable

import cv2
import nltk
import numpy as np
import pandas as pd
import SimpleITK as sitk
from dragon_baseline.main import DragonBaseline
from dragon_baseline.nlp_algorithm import TaskDetails
from tqdm import tqdm

from vision.pathology.info import image_info
from vision.pathology.utils import select_coordinates_with_tissue
from vision.pathology.wsi import FilterParams, TilingParams, WholeSlideImage
from vision.radiology.patch_extraction import extract_patches

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")


def load_task_description(input_path: Path = Path("/input/unicorn-task-description.json")) -> dict[str, str]:
    """
    Read information from unicorn-task-description.json
    """
    with open(input_path, "r") as f:
        task_description = json.load(f)
    return task_description


def load_inputs(input_path: Path = Path("/input/inputs.json")) -> list[dict[str, Any]]:
    """
    Read information from inputs.json
    """
    input_information_path = Path(input_path)
    with input_information_path.open("r") as f:
        input_information = json.load(f)

    for item in input_information:
        relative_path = item["interface"]["relative_path"]
        item["input_location"] = Path(f"/input/{relative_path}")

    return input_information


def sanitize_json_content(obj):
    if isinstance(obj, dict):
        return {k: sanitize_json_content(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        return [sanitize_json_content(v) for v in obj]
    elif isinstance(obj, (str, int, bool, float)):
        return obj
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    else:
        return obj.__repr__()


def write_json_file(*, location, content):
    # Writes a json file with the sanitized content
    content = sanitize_json_content(content)
    with open(location, "w") as f:
        f.write(json.dumps(content, indent=4))


def resolve_image_path(*, location: str | Path) -> Path:
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    if len(input_files) != 1:
        raise ValueError(f"Expected one image file, got {len(input_files)}")

    input_file = Path(input_files[0])
    return input_file


def feature_extraction(image: np.ndarray) -> list[float]:
    # Extract features from a patch
    checksum = hashlib.md5(image.tobytes()).hexdigest()
    checksum = [int(c, 16) for c in checksum]
    image_features = [np.mean(image), np.std(image), np.median(image), np.min(image), np.max(image)]
    features = image_features + checksum
    features = [float(f) for f in features]
    return features


def aggregate_patch_level_neural_representations(
    patch_level_neural_representations: list[dict],
):
    # Aggregate the patch-level neural representations
    aggregated_neural_representation = {
        "title": patch_level_neural_representations["title"],
        "features": list(np.mean(np.array([patch["features"] for patch in patch_level_neural_representations["patches"]]), axis=0)),
    }
    return aggregated_neural_representation


def make_patch_level_neural_representation(
    *,
    title: str,
    patch_features: Iterable[dict],
    patch_size: Iterable[int],
    patch_spacing: Iterable[float],
    image_size: Iterable[int],
    image_spacing: Iterable[float],
    image_origin: Iterable[float] = None,
    image_direction: Iterable[float] = None,
) -> dict:
    if image_origin is None:
        image_origin = [0.0] * len(image_size)
    if image_direction is None:
        image_direction = np.identity(len(image_size)).flatten().tolist()
    return {
        "meta": {
            "patch-size": list(patch_size),
            "patch-spacing": list(patch_spacing),
            "image-size": list(image_size),
            "image-origin": list(image_origin),
            "image-spacing": list(image_spacing),
            "image-direction": list(image_direction),
        },
        "patches": list(patch_features),
        "title": title,
    }


def process_image_pathology(
    image_path: Path,
    tissue_mask_path: Path,
    tiling_params: TilingParams,
    filter_params: FilterParams,
    title: str = "patch-level-neural-representation",
    max_number_of_tiles: int | None = None,
    num_workers: int = 8,
    seed: int | None = 576,
) -> list[dict]:

    """
    Generate a list of patch features from a pathology image

    Args:
    """
    # show image information
    image_info(image_path=image_path)

    
    patch_features = []
    wsi = WholeSlideImage(image_path, tissue_mask_path)
    coordinates, tissue_percentages, patch_level, resize_factor, _, = wsi.get_tile_coordinates(
        tiling_params=tiling_params,
        filter_params=filter_params,
        num_workers=num_workers,
    )

    patch_coordinates, _ = select_coordinates_with_tissue(
        coordinates=coordinates,
        tissue_percentages=tissue_percentages,
        max_number_of_tiles=max_number_of_tiles,
        seed=seed,
    )

    print(f"Extracting features from patches")
    for x, y in tqdm(patch_coordinates, desc="Extracting features"):
        patch_spacing = wsi.spacings[patch_level]
        patch_size_resized = int(tiling_params.tile_size * resize_factor)
        patch = wsi.get_tile(x, y, [patch_size_resized, patch_size_resized], patch_spacing)
        # resize patch to the desired patch size
        patch = cv2.resize(patch, (tiling_params.tile_size, tiling_params.tile_size), interpolation=cv2.INTER_LINEAR)
        features = feature_extraction(patch)
        patch_features.append({
            "coordinates": (x, y),
            "features": features,
        })

    patch_level_neural_representation = make_patch_level_neural_representation(
        title=title,
        patch_features=patch_features,
        patch_size=[tiling_params.tile_size, tiling_params.tile_size],
        patch_spacing=[tiling_params.spacing, tiling_params.spacing],
        image_size=wsi.level_dimensions[0],
        image_spacing=[wsi.spacings[0], wsi.spacings[0]],
    )

    return patch_level_neural_representation


def process_image_radiology(
    image_path: Path,
    title: str = "patch-level-neural-representation",
    patch_size: list[int] = [224, 224, 16],
    patch_spacing: list[float] | None = None,
) -> list[dict]:
    """
    Generate a list of patch features from a radiology image

    Args:
        image_path (Path): Path to the image file
        title (str): Title of the patch-level neural representation
        patch_size (list[int]): Size of the patches to extract
        patch_spacing (list[float] | None): Voxel spacing of the image. If specified, the image will be resampled to this spacing before patch extraction.
    Returns:
        list[dict]: List of dictionaries containing the patch features
        - coordinates (list[tuple]): List of coordinates for each patch, formatted as:
            ((x_start, x_end), (y_start, y_end), (z_start, z_end)).
        - features (list[float]): List of features extracted from the patch
    """
    patch_features = []
    print(f"Reading image from {image_path}")
    image = sitk.ReadImage(str(image_path))

    print(f"Extracting patches from image")
    patches, coordinates = extract_patches(
        image=image,
        patch_size=patch_size,
        spacing=patch_spacing,
    )
    if patch_spacing is None:
        patch_spacing = image.GetSpacing()

    print(f"Extracting features from patches")
    for patch, coordinates in tqdm(zip(patches, coordinates), total=len(patches), desc="Extracting features"):
        patch_array = sitk.GetArrayFromImage(patch)
        features = feature_extraction(patch_array)
        patch_features.append({
            "coordinates": coordinates[0],  # save the start coordinates
            "features": features,
        })

    patch_level_neural_representation = make_patch_level_neural_representation(
        patch_features=patch_features,
        patch_size=patch_size,
        patch_spacing=patch_spacing,
        image_size=image.GetSize(),
        image_origin=image.GetOrigin(),
        image_spacing=image.GetSpacing(),
        image_direction=image.GetDirection(),
        title=title,
    )

    return patch_level_neural_representation


def process_image(
    image_path: Path,
    tissue_mask_path: Path,
    task_description: dict[str, str],
    title: str = "patch-level-neural-representation",
) -> list[dict]:
    """
    Generate a list of patch features from an image
    """
    if task_description["domain"] == "pathology":
        task_type = task_description["task_type"]
        max_number_of_tiles = 14_000 if task_type in ["classification", "regression"] else None

        if task_type in ["detection", "segmentation"]:
            tiling_params = TilingParams(
                spacing=0.5, tolerance=0.07, tile_size=224, overlap=0.0,
                drop_holes=False, min_tissue_ratio=0.1, use_padding=True
            )
            filter_params = FilterParams(
                ref_tile_size=64, a_t=1, a_h=1, max_n_holes=8
            )
        else:
            tiling_params = TilingParams(
                spacing=0.5, tolerance=0.07, tile_size=512, overlap=0.0,
                drop_holes=False, min_tissue_ratio=0.25, use_padding=True
            )
            filter_params = FilterParams(
                ref_tile_size=256, a_t=4, a_h=2, max_n_holes=8
            )

        patch_level_neural_representation = process_image_pathology(
            image_path=image_path,
            tissue_mask_path=tissue_mask_path,
            title=title,
            max_number_of_tiles=max_number_of_tiles,
            tiling_params=tiling_params,
            filter_params=filter_params,
        )
    elif task_description["domain"] in ["CT", "MR"]:
        patch_level_neural_representation = process_image_radiology(
            image_path=image_path,
            title=title,
        )
    else:
        warnings.warn(f"No processing implemented for domain: {task_description['domain']}, using default: radiology")
        patch_level_neural_representation = process_image_radiology(
            image_path=image_path,
        )

    return patch_level_neural_representation


def split_into_words(text: str | list[str]) -> list[str]:
    """
    Split text into words
    """
    if isinstance(text, str):
        return text.split()
    elif isinstance(text, list):
        return [word for sentence in text for word in sentence.split()]
    else:
        raise ValueError(f"Expected str or list[str], got {type(text)}")


def predict_language(
    task_config: "TaskDetails",
    few_shots: pd.DataFrame,
    test_case: pd.Series,
) -> dict:
    """"
    Make a prediction for a test case (given a task configuration and few-shots)
    """
    # Make a prediction for a test case in any way you'd like
    # In this example, we'll return the label of the most similar few-shot
    test_input: str = test_case[task_config.input_name]

    results = {}
    for i, shot in few_shots.iterrows():
        # Calculate the edit distance between the test input and the few-shot input (at word level)
        distance = nltk.edit_distance(
            split_into_words(test_input),
            split_into_words(shot[task_config.input_name]),
        )
        results[distance] = shot[task_config.target.label_name]

    # Return the label with least distance
    min_distance = min(results.keys())
    prediction = results[min_distance]
    return {
        "uid": test_case["uid"],
        task_config.target.prediction_name: prediction,
    }


def run_vision_and_visionlanguage():
    # Read the task description and inputs information
    task_description = load_task_description(input_path=INPUT_PATH / "unicorn-task-description.json")
    print(f"Task description: {task_description}")

    input_information = load_inputs(input_path=INPUT_PATH / "inputs.json")
    print(f"Input information: {input_information}")

    # Read the input
    image_inputs = []
    tissue_mask_path = None
    for input_socket in input_information:
        if input_socket["interface"]["kind"] == "Image":
            image_inputs.append(input_socket)
        elif input_socket["interface"]["kind"] == "Segmentation":
            tissue_mask_path = resolve_image_path(location=input_socket["input_location"])

    print(f"Have {len(image_inputs)} image(s) {'and its tissue mask' if tissue_mask_path is not None else ''}")

    # Some additional resources might be required, include these in one of two ways.

    # Option 1: part of the Docker-container image: resources/
    resource_dir = Path("/opt/app/resources")
    with open(resource_dir / "some_resource.txt", "r") as f:
        print(f.read())

    # Option 2: upload them as a separate tarball to Grand Challenge (go to your Algorithm > Models). The resources in the tarball will be extracted to `model_dir` at runtime.
    model_dir = Path("/opt/ml/model")
    with open(
        model_dir / "a_tarball_subdirectory" / "some_tarball_resource.txt", "r"
    ) as f:
        print(f.read())

    # Process the inputs: any way you'd like
    neural_representations = []
    for image_input in image_inputs:
        neural_representation = process_image(
            image_path=resolve_image_path(location=image_input["input_location"]),
            tissue_mask_path=tissue_mask_path,
            task_description=task_description,
            title=image_input["interface"]["slug"]
        )
        neural_representations.append(neural_representation)

    if task_description["task_type"] in ["classification", "regression"]:
        # Aggregate the patch-level neural representations for classification and regression
        output = [
            aggregate_patch_level_neural_representations(
                patch_level_neural_representations=patch_level_neural_representations
            )
            for patch_level_neural_representations in neural_representations
        ]
        output_path = OUTPUT_PATH / "image-neural-representation.json"
    elif task_description["modality"] == "vision-language":
        # Generate a description of the image any way'd like!
        neural_representations = [
            aggregate_patch_level_neural_representations(
                patch_level_neural_representations=patch_level_neural_representations
            )
            for patch_level_neural_representations in neural_representations
        ]

        output = []
        for input_socket, neural_representation in zip(input_information, neural_representations):
            uid = input_socket["image"]["pk"]
            if neural_representation["features"][0] > neural_representation["features"][2]:
                output.append({"uid": uid, "text": "The image has a higher average patch-wise mean intensity than average patch-wise median intensity"})
            else:
                output.append({"uid": uid, "text": "The image has a lower average patch-wise mean intensity than average patch-wise median intensity"})
        output_path = OUTPUT_PATH / "nlp-predictions-dataset.json"
    else:
        output = neural_representations
        output_path = OUTPUT_PATH / "patch-neural-representation.json"

    # Save your output
    write_json_file(
        location=output_path,
        content=output,
    )

    print(f"Saved output to {output_path}")
    return 0


def run_language():
    # Read the task configuration, few-shots and test data
    # We'll leverage the DRAGON baseline algorithm for this
    algorithm = DragonBaseline()
    algorithm.load()
    algorithm.analyze()  # needed for verification of predictions
    task_config = algorithm.task
    few_shots = algorithm.df_train
    test_data = algorithm.df_test
    print(f"Task description: {task_config}")

    # Make predictions for the test data
    predictions = []
    for i, test_case in tqdm(test_data.iterrows(), total=len(test_data), desc="Predicting"):
        prediction = predict_language(
            task_config=task_config,
            few_shots=few_shots,
            test_case=test_case,
        )
        predictions.append(prediction)

    # Cast the predictions to the correct type (if needed)
    predictions = algorithm.cast_predictions(predictions)

    # Save the predictions
    test_predictions_path = OUTPUT_PATH / "nlp-predictions-dataset.json"
    write_json_file(
        location=test_predictions_path,
        content=predictions,
    )

    # Verify the predictions
    algorithm.test_predictions_path = test_predictions_path
    algorithm.verify_predictions()

    print(f"Saved neural representation to {test_predictions_path}")
    return 0


def run():
    # check if the task is image or text
    if (INPUT_PATH / "nlp-task-configuration.json").exists():
        return run_language()
    else:
        return run_vision_and_visionlanguage()


if __name__ == "__main__":
    raise SystemExit(run())
