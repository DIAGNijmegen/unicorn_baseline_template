import json
from pathlib import Path


def sort_coordinates_with_tissue(coords, tissue_percentages):
    # mock region filenames
    mocked_filenames = [f"{x}_{y}.jpg" for x, y in coords]
    # combine mocked filenames with coordinates and tissue percentages
    combined = list(zip(mocked_filenames, coords, tissue_percentages))
    # sort combined list by mocked filenames
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # extract sorted coordinates and tissue percentages
    sorted_coords = [coord for _, coord, _ in sorted_combined]
    sorted_tissue_percentages = [tissue for _, _, tissue in sorted_combined]
    return sorted_coords, sorted_tissue_percentages


def save_feature_to_json(feature_vector):
    """
    Saves the extracted feature vector to a JSON file in the required format.
    """
    output_dict = [
        {"title": "images/prostate-tissue-biopsy-wsi", "features": feature_vector}
    ]
    output_path = Path("/output")
    output_filename = output_path / "image-neural-representation.json"
    with open(output_filename, "w") as f:
        json.dump(output_dict, f, indent=4)

    print(f"Feature vector saved to {output_filename}")
