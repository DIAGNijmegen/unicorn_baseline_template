1. Clone this repository:
   ```bash
   git clone git@github.com:DIAGNijmegen/unicorn_baseline_template.git
   cd unicorn_baseline_template
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv unicorn
   source unicorn/bin/activate
   ```

3. Install ASAP 2.2

   - download ASAP
      ```bash
      curl -L "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.2-(Nightly)/ASAP-2.2-Ubuntu2204.deb" -o /tmp/ASAP.deb
      ```
   - install ASAP
      ```bash
      sudo apt-get install --assume-yes /tmp/ASAP.deb
      ```
   - add ASAP binary to your virtual environment's `PYTHONPATH`:
      ```bash
      export SITE_PACKAGES=`unicorn/bin/python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"`
      echo "/opt/ASAP/bin" | sudo tee ${SITE_PACKAGES}/asap.pth
      ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the inference script locally:
   ```bash
   python inference.py
   ```
