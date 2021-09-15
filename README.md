[![PyPI version](https://badge.fury.io/py/west-ic-antenna.svg)](https://badge.fury.io/py/west-ic-antenna)

# WEST ICRH Antenna Electrical Model

This code is a numerical model of the WEST Ion Cyclotron Resonance Antenna. The code builds the electrical circuit of an antenna, connecting the different parts of the antenna separately full-wave calculated. The front-face of the antenna, evaluated using bespoke antenna code or full-wave software, is given as an input. The antenna tuning capacitors can be automatically optimized for a given frequency. The voltages and currents at the capacitors can be evaluated for a given antenna excitation (power and phase).   


## Antenna Digital Twin

You can test the antenna model online with Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jhillairet/WEST_IC_antenna/HEAD?filepath=doc%2Fdigital_twin.ipynb)

## Installation

The package can be installed via pypi:

`pip install west-ic-antenna`

## References
- Hillairet, J., 2020. RF network analysis of the WEST ICRH antenna with the open-source python scikit-RF package. AIP Conference Proceedings 2254, 070010. https://doi.org/10/ghbw5p (free version https://hal-cea.archives-ouvertes.fr/cea-02459571/document)
