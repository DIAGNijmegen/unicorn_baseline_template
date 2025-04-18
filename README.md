# 🧱 UNICORN Baseline Template

This repository provides a **minimal working template** for participating in the [UNICORN Challenge](https://unicorn-challenge.ai).<br>
It serves as a **starting point** for your own submission and implements the required boilerplate to run across all tasks in the challenge.

## 🔍 What It Does

This repository:
- Reads the challenge input for each task (e.g., image paths, report texts, etc.).
- Produces an output JSON file for each sample.
- Provides a dummy checksum-based feature extraction as a placeholder for real models for pathology and radiology vision tasks.
- Provides a dummy checksum-based feature extraction as a placeholder for real models.

📦 This ensures your code is **compatible with the challenge evaluation pipeline**, even before you plug in a real model!

## 📁 Structure

- `inference.py`: Main entry point for processing inputs and generating outputs.
- `vision/`: Contains modules for encoding and processing pathology and radiology images.
  - `pathology/`: Utilities and classes for handling pathology images.
    - `wsi.py`: Handles whole slide images and tile extraction.
    - `wsi_utils.py`: Utilities for tissue detection and contour processing.
    - `info.py`: Displays information about multi-resolution images.
    - `utils.py`: Helper functions for sorting coordinates and saving features.
  - `radiology/`: Utilities for handling radiology images.
    - `patch_extraction.py`: Extracts patches from 3D radiology images.
- `model/`: Placeholder for model-related resources.
  - `README.md`: Instructions for uploading or including models.
  - `a_tarball_subdirectory/`: Example subdirectory for tarball resources.
- `requirements.in` and `requirements.txt`: Python dependencies for the project.
- `Dockerfile`: Defines the container environment for running the algorithm.
- `do_build.sh`: Script to build the Docker container.
- `do_test_run.sh`: Script to test the container locally.
- `do_save.sh`: Script to save the container image and optional tarball for upload.
- `.gitignore`: Specifies files and directories to ignore in version control.

## 🚀 Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/your-org/unicorn_baseline_template.git
   cd unicorn_baseline_template
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the inference script locally:
   ```bash
   python inference.py
   ```

4. Build and test the Docker container:
   ```bash
   ./do_build.sh
   ./do_test_run.sh
   ```

5. Save the container for upload:
   ```bash
   ./do_save.sh
   ```

## 🛠️ Customization

- Modify `inference.py` to implement your own feature extraction or prediction logic.
- Add your model weights to the `model/` directory or upload them as a tarball to Grand Challenge.
- Update `requirements.in` to include additional Python dependencies and regenerate `requirements.txt` using `pip-compile`.

## 📄 License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.