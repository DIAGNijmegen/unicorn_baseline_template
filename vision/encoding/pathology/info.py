import argparse
from pathlib import Path
import multiresolutionimageinterface as mir


def image_info(image_path: Path | str, verbose: bool = True) -> None:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise ValueError(f'Invalid image path: {image_path}')

    reader = mir.MultiResolutionImageReader()
    image = reader.open(str(image_path))

    num_levels = image.getNumberOfLevels()
    num_channels = image.getSamplesPerPixel()
    dtype = image.getDataType()

    shapes = [
        tuple(reversed(image.getLevelDimensions(level)))
        for level in range(num_levels)
    ]
    downsamplings = [image.getLevelDownsample(level) for level in range(num_levels)]

    spacing_base = image.getSpacing()
    spacings = [
        spacing_base[0] * downsampling if spacing_base else None
        for downsampling in downsamplings
    ]

    print(f'Number of levels: {num_levels}')
    print(f'Number of channels: {num_channels}')
    print(f'Data type: {dtype}')

    for level in range(num_levels):
        shape = shapes[level]
        spacing = spacings[level]
        downsampling = downsamplings[level]

        if verbose:
            print(f'Level {level}:')
            print(f'    Shape: {shape}')
            print(f'    Spacing: {spacing} um' if spacing is not None else '    Spacing: Unknown')
            print(f'    Downsampling: {downsampling}x')
        else:
            spacing_str = f'{spacing} um' if spacing is not None else 'Unknown'
            print(f'[{level}] {shape}; {spacing_str}; {downsampling}x')


def main():
    parser = argparse.ArgumentParser(
        description='Display information about a multi-resolution image.'
    )
    parser.add_argument('image_path', type=str, help='Path to the multi-resolution image')
    parser.add_argument('--verbose', action='store_true', help='Display verbose information')

    args = parser.parse_args()
    image_info(args.image_path, args.verbose)

if __name__ == '__main__':
    main()
