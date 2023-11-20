<!--
Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# QNTRPO: Quasi-Newton Trust Region Policy Optimization

System requirements:
The code has been tested on these environments.

1. Ubuntu 16.04 LTS
2. Python 3.6.7 (will not work on Python 3.6.0 due to some issues of pytorch and python 3.6.0)
3. Torch 1.1.0 (the most recent version of pytorch will work )
4. Mujoco_py==1.50
5. Gym

## Features

QNTRPO solves the Policy Optimization problem that arises in Reinforcement Learning using a Quasi-Newton Trust Region algorithm.

## Installation

The code depends on external libraries. Install the software following the instructions below. We are describing the installation in a virtual environment.
```
conda create -n qntrpo python=3.11 anaconda

source activate qntrpo

conda install pytorch
```

Install Mujoco and mujoco-py following the instructions in https://github.com/openai/mujoco-py (License: `MIT`)

Install Gym following the instructions in https://github.com/openai/gym (License: `MIT`)

## Usage

If a user wants to change the trust region radius for optimization, they should change the parameter "tr_maxdelta" on line 67 in the code "trust_region_opt_torch.py". The current value is 1e-1. It is suggested to run the code with this value. The performance of the algorithm on other values have not been fully tested yet.

 A different batch size could be used by adding another argument while calling the code, --batch-size N, where (N is an integer say 25000), i.e.,

 ```
python main.py --env-name "Walker2d-v2" --seed 1243 --batch-size 25000
```

## Testing

QNTRPO algorithm can be tested by running the following in a terminal (for example for Walker2d and seed, say 1243).
```
python main.py --env-name "Walker2d-v2" --seed 1243
```

## Citation

If you use the software, please cite the following  ([TR2019-120](https://www.merl.com/publications/TR2019-120)):

```bibTeX
@inproceedings{Jha2019oct,
author = {Jha, Devesh K. and Raghunathan, Arvind and Romeres, Diego},
title = {Quasi-Newton Trust Region Policy Optimization},
booktitle = {Conference on Robot Learning (CoRL)},
year = 2019,
editor = {Leslie Pack Kaelbling and Danica Kragic and Komei Sugiura},
pages = {945--954},
month = oct,
publisher = {Proceedings of Machine Learning Research},
url = {https://www.merl.com/publications/TR2019-120}
}
```

## Contact

Please contact one of us Devesh K Jha (jha@merl.com), Arvind U Raghunathan (raghunathan@merl.com), or Diego Romeres (romeres@merl.com).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:

```
Copyright (C) 2019, 2023 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
