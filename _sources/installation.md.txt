## Installation

### Using the PIP package

A PIP package is available on pypi and can be installed with:

```bash
pip install thermontfa
```

If you want to install optional dependencies for development:

```bash
pip install thermontfa[dev]
```

### From this repository

The most recent version of the PIP package can also be installed directly after cloning this repository.

```bash
git clone https://github.com/DataAnalyticsEngineering/ThermoNTFA.git
cd ThermoNTFA
pip install -e .
```

If you want to install optional dependencies for development:

```bash
git clone https://github.com/DataAnalyticsEngineering/ThermoNTFA.git
cd ThermoNTFA
pip install -e .[dev]
```

### Requirements

TODO: update dependencies

TODO: upload dataset to Darus

TODO: provide functionality for download from Darus

- Python 3.9 or later
- `numpy` and `h5py` (installed as part of the `thermontfa` PIP package)
- Input
  dataset: [![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--2822-d45815.svg)](https://doi.org/10.18419/darus-2822)

All necessary data can be downloaded from [DaRUS](https://darus.uni-stuttgart.de/) using the script [`download_data.sh`](download_data.sh).
