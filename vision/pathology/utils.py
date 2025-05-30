import json
import random
from pathlib import Path


def select_coordinates_with_tissue(
    *,
    coordinates: list[tuple],
    tissue_percentages: list[float],
    max_number_of_tiles: int | None = None,
    seed: int | None = None,
) -> tuple[list[tuple], list[float]]:
    """
    Select coordinates and their corresponding tissue percentages based on an
    optional maximum number of tiles. If more than the maximum number of tiles
    are found with full tissue content (100%), a random selection of tiles is
    made. Otherwise, the tiles are sorted by tissue percentage and the top tiles
    are selected.

    Args:
        coordinates (list of tuple): A list of tuples representing coordinates,
            where each tuple contains two integers (x, y).
        tissue_percentages (list of float): A list of tissue percentages
            corresponding to the tile for each coordinate.
        max_number_of_tiles (int | None): The maximum number of tiles to select.
            If None, all tiles above the minimum tissue percentage will be selected.

    Returns:
        tuple: A tuple containing two lists:
            - sorted_coordinates (list of tuple): The coordinates sorted based
              on the tissue percentages.
            - sorted_tissue_percentages (list of float): The tissue
              corresponding to the sorted coordinates.
    """

    # Separate perfect tissue tiles
    perfect = [(coord, perc) for coord, perc in zip(coordinates, tissue_percentages) if perc == 1.0]
    if max_number_of_tiles is not None and len(perfect) > max_number_of_tiles:
        rng = random.Random(seed)
        selected = rng.sample(perfect, max_number_of_tiles)
    else:
        # Sort by descending tissue percentage and take top N if needed
        all = [(coord, perc) for coord, perc in zip(coordinates, tissue_percentages)]
        all.sort(key=lambda x: x[1], reverse=True)
        selected = all[:max_number_of_tiles] if max_number_of_tiles is not None else all

    selected_coordinates, selected_percentages = zip(*selected)
    return list(selected_coordinates), list(selected_percentages)


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
