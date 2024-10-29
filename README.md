# ThermoNTFA

<!-- The badges we want to display -->
[![arXiv][arxiv-shield]][arxiv-url]
[![DOI][doi-shield]][doi-url]
[![Documentation Status][docs-shield]][docs-url]
[![pytest](https://github.com/DataAnalyticsEngineering/ThermoNTFA/actions/workflows/ci.yml/badge.svg)](https://github.com/DataAnalyticsEngineering/ThermoNTFA/actions/workflows/ci.yml)
[![flake8](https://img.shields.io/badge/flake8-checked-blue.svg)](https://flake8.pycqa.org/)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Python](https://img.shields.io/badge/python-3.11-purple.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/thermontfa)](https://pypi.org/project/thermontfa/)
<!--[![MIT License][license-shield]][license-url]-->
<!--[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE.md)-->


This repository contains a material routine written in *Python* for the thermo-elasto-plastic Nonuniform Transformation Field Analysis (**ThermoNTFA**) published in _Computer Methods in Applied Mechanics and Engineering_:

Fritzen, Felix and Herb, Julius and Sharba, Shadi, Thermo-Plastic Nonuniform Transformation Field Analysis for Eigenstress Analysis of Materials Undergoing Laser Melt Injection. Computer Methods in Applied Mechanics and Engineering (2024). DOI: [10.1016/j.cma.2024.117487](https://doi.org/10.1016/j.cma.2024.117487)

## Abstract

In engineering applications, surface modifications of materials can greatly influence the lifetime of parts and structures. For instance, laser melt injection (LMI) of ceramic particles into a metallic substrate can greatly improve abrasive resistance. The LMI process is challenging to model due to the rapid temperature changes, which induce high mechanical stresses. Ultimately, this leads to plastification and residual eigenstresses in particles and matrix. These depend on the process parameters. In order to predict these stresses, we propose a major extension of the Nonuniform Transformation Field Analysis that enables the method to cope with strongly varying thermo-elastic material parameters over a large temperature range (here: 300 to 1300K). The newly proposed $\theta$-NTFA method combines the NTFA with a Galerkin projection to solve for the self-equilibrated fields needed to gain the NTFA system matrices. For that, we exploit our recent thermo-elastic reduced order model [1] and extend it to allow for arbitrary polarization strains.  An efficient implementation and a rigorous separation of the derivation of the reduced order model is proposed. The new $\theta$-NTFA is then validated for various thermo-mechanical loadings and in thermo-mechanical two-scale simulations.

[1] S. Sharba, J. Herb, F. Fritzen, Reduced order homogenization of thermoelastic materials with strong temperature
dependence and comparison to a machine-learned model, Archive of Applied Mechanics 93 (7) (2023) 2855–2876.
doi: [10.1007/s00419-023-02411-6](https://doi.org/10.1007/s00419-023-02411-6)

## Documentation

The documentation of this software, including examples on how to use **ThermoNTFA**, can be found under [Documentation](https://DataAnalyticsEngineering.github.io/ThermoNTFA/).

## Features

- The **ThermoNTFA** acts as a reduced order model (ROM) and approximates the effective behavior of composite materials that consist of thermoelastic and thermoplastic constituents.
- The material parameters of all constituents are allowed to depend strongly on the temperature.
- This temperature-dependence is reflected in the ROM that is based on interpolated space-saving tabular data at arbitrarily many temperature points.
- Possible application: Eigenstress Analysis of Laser Dispersed Materials

## Workflow

![thermoNTFA](https://github.com/DataAnalyticsEngineering/ThermoNTFA/blob/main/docs/images/ntfa_workflow.jpg?raw=True "workflow")

### Offline phase: Training of the thermo-mechanical NTFA

1. Generate data using thermo-elasto-plastic simulations on the microscale at select temperatures $`\theta \in \mathcal{T}`$.
For that, we used an in-house FE solver. However, any other suitable simulation software could be used.

2. Compute a reduced basis consisting of plastic modes $`\underline{\underline{P}}_{\mathsf{p}}`$ via snapshot POD of the simulated plastic strain fields $`\{\varepsilon_\mathsf{p}(x | \theta)\}^n_{i=1}`$.
The corresponding implementation is available in our [AdaptiveThermoMechROM](https://github.com/DataAnalyticsEngineering/AdaptiveThermoMechROM) repository in the module [ntfa.py](https://github.com/DataAnalyticsEngineering/AdaptiveThermoMechROM/blob/ntfa/ntfa.py).

3. Perform additional linear-elastic simulations to determined self-equilibrated fields for the plastic modes $`\underline{\underline{P}}_\mathsf{p}`$ at select temperatures $`\theta \in \mathcal{T}`$.
Again, we used our in-house FE solver for that.

4. Based on the generated data at select temperatures $`\theta \in \mathcal{T}`$ we perform an interpolation to arbitrarily many intermediate temperatures $`\theta_j`$.
This method is published in our paper ["Reduced order homogenization of thermoelastic materials with strong temperature-dependence and comparison to a machine-learned model"](https://doi.org/10.1007/s00419-023-02411-6), where we show that it produces highly accurate results while the effort is almost on par with linear interpolation.

5. Using the interpolated data, the NTFA system matrices $`\underline{\underline{A}}(\theta_j)`$, $`\underline{\underline{D}}(\theta_j)`$, $`\bar{\underline{\underline{C}}}(\theta_j)`$, $`\underline{\tau}_{\mathrm{\theta}}(\theta_j)`$, and $`\underline{\tau}_{\mathsf{p}}(\theta_j)`$ are computed and stored as tabular data.
The corresponding implementation is available in our [AdaptiveThermoMechROM](https://github.com/DataAnalyticsEngineering/AdaptiveThermoMechROM) repository in the module [ntfa.py](https://github.com/DataAnalyticsEngineering/AdaptiveThermoMechROM/blob/ntfa/ntfa.py).

### Online phase: Usage of the thermo-mechanical NTFA in simulations on the macroscale

1. Load the tabular data for the NTFA matrices $`\underline{\underline{A}}(\theta_j)`$, $`\underline{\underline{D}}(\theta_j)`$, $`\bar{\underline{\underline{C}}}(\theta_j)`$, $`\underline{\tau}_{\mathrm{\theta}}(\theta_j)`$, and $`\underline{\tau}_{\mathsf{p}}(\theta_j)`$ that are generated in the offline phase based on direct numerical simulations on the microscale.
Optionally truncate the NTFA modes $`N_{\mathrm{modes}}`$ to be used.

2. Perform a linear interpolation to determine the NTFA matrices at the current model temperature based on the tabular data.
Given that the tabular data is available at sufficiently many temperatures, the linear interpolation provides results with decent accuracy.
This is done using the class [`thermontfa.TabularInterpolation`](https://github.com/DataAnalyticsEngineering/ThermoNTFA/blob/main/thermontfa/tabular_interpolation.py).

3. Use the tabular data to initialize the thermo-mechanical NTFA UMAT that is implemented in the class [`thermontfa.ThermoMechNTFA`](https://github.com/DataAnalyticsEngineering/ThermoNTFA/blob/main/thermontfa/thermoNTFA.py).
This reference implementation in Python can be transcribed to an UMAT for other academic or commercial simulation softwares.
The numerical experiments in our paper are conducted using an implementation of the thermo-mechanical NTFA UMAT in C++ for an in-house FE solver.

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

- Python 3.10 or later
- `numpy` and `h5py` (installed as part of the `thermontfa` PIP package)
- Supplemental data: [![Identifier](https://img.shields.io/badge/doi-10.18419%2Fdarus--4509-d45815.svg)](https://doi.org/10.18419/darus-4509)

All necessary data can be downloaded from [DaRUS](https://darus.uni-stuttgart.de/) using the script [`download_data.sh`](download_data.sh).

## Manuscript

Fritzen, Felix and Herb, Julius and Sharba, Shadi, Thermo-Plastic Nonuniform Transformation Field Analysis for Eigenstress Analysis of Materials Undergoing Laser Melt Injection. Computer Methods in Applied Mechanics and Engineering (2024). DOI: [10.1016/j.cma.2024.117487](https://doi.org/10.1016/j.cma.2024.117487)

Affiliation: [Data Analytics in Engineering, University of Stuttgart](http://www.mib.uni-stuttgart.de/dae)

## Acknowledgments

- The IGF-Project with the IGF-No.: 21079 N / DVS-No.: 06.3341 of the “Forschungsvereinigung Schweißen und verwandte Verfahren e.
  V.” of the German Welding Society (DVS), Aachener Str. 172, 40223 Düsseldorf was funded by the Federal Ministry for Economic
  Affairs and Climate Action (BMWK) via the German Federation of Industrial Research Associations (AiF) in accordance with the
  policy to support the Industrial Collective Research (IGF) on the basis of a decision by the German Bundestag. Furthermore, the
  authors gratefully acknowledge the collaboration with the members of the project affiliated committee regarding the support of
  knowledge, material and equipment over the course of the research.

- Contributions by Felix Fritzen are partially funded by Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under
  Germany’s Excellence Strategy - EXC 2075 – 390740016. Felix Fritzen is funded by Deutsche Forschungsgemeinschaft (DFG, German
  Research Foundation) within the Heisenberg program DFG-FR2702/8 - 406068690 and DFG-FR2702/10 - 517847245.

- Contributions of Julius Herb are partially funded by the Ministry of Science, Research and the Arts (MWK) Baden-Württemberg, Germany, within the Artificial Intelligence Software Academy (AISA).

- The authors acknowledge the support by the Stuttgart Center for Simulation Science (SimTech).

[license-shield]: https://img.shields.io/github/license/DataAnalyticsEngineering/ThermoNTFA.svg
[license-url]: https://github.com/DataAnalyticsEngineering/ThermoNTFA/blob/main/LICENSE
[doi-shield]: https://img.shields.io/badge/doi-10.18419%2Fdarus--4509-d45815.svg
[doi-url]: https://doi.org/10.18419/darus-4509
[arxiv-shield]: https://img.shields.io/badge/SSRN-4926472-b31b1b.svg
[arxiv-url]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4926472
[docs-url]: https://DataAnalyticsEngineering.github.io/ThermoNTFA
[docs-shield]: https://img.shields.io/badge/docs-online-blue.svg
