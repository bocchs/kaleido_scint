# Kaleidoscopic Scintillation Event Imaging

***CVPR 2026***<br>
[https://arxiv.org/abs/2512.03216](https://arxiv.org/abs/2512.03216)

## Install 

```bash
# clone repo
git clone https://github.com/bocchs/kaleido_scint.git
cd kaleido_scint

# create virtual env
python -m venv .venv
source .venv/bin/activate

# install requirements
pip install -r requirements.txt
```

## Running code
To change parameters, modify `init_values()` in `code/run.py`.

```bash
cd code
python run.py
```

## Data
`images` contains all experimental images that have at least 60 counts.
`images_5_components` contains all experimental images that were detected to contain the event and four mirror reflections.