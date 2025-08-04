# UMPicker
## Environment Requirements
Since this project relies on the [mamba-ssm](https://github.com/state-spaces/mamba) package, this project needs to be run on a Linux host with an NVIDIA GPU.
Other requirements:
* PyTorch 1.12+
* CUDA 11.6+

## Installation
### 1. Creation of a new virtual environment
This project requires a conda virtual environment with Python 3.10. Use the following command to create a new conda environment named umpicker:
```
conda create -n umpicker python=3.10
```
### 2. Activation of the environment
```
conda activate umpicker
```
### 3. Installation of dependencies
```
pip install -r requirements.txt
```
### 4. Installation of mamba-ssm
For detailed installation steps of mamba-ssm, please refer to the instructions of https://github.com/state-spaces/mamba and install version 2.2.2 of mamba-ssm to meet the requirements of this environment.

## Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please submit an issue or pull request.

## References
Dao, T., Gu, A., 2024. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. arXiv preprint arXiv:2405.21060.

Woollam, J., Münchmeyer, J., Tilmann, F., Rietbrock, A., Lange, D., Bornstein, T., Diehl, T., Giunchi, C., Haslinger, F., Jozinović, D., et al., 2022. Seisbench—a toolbox for machine learning in seismology. Seismological Society of America 93, 1695–1709.

Münchmeyer, J., Woollam, J., Rietbrock, A., Tilmann, F., Lange, D., Bornstein, T., Diehl, T., Giunchi, C., Haslinger, F., Jozinović, D., et al., 2022. Which picker fits my data? a quantitative evaluation of deep learning based seismic pickers. Journal of Geophysical Research: Solid Earth 127, e2021JB023499.

Mousavi, S.M., Sheng, Y., Zhu, W., Beroza, G.C., 2019a. Stanford earthquake dataset (stead): A global data set of seismic signals for ai. IEEE Access 7, 179464–179476.

Zhao, M., Xiao, Z., Chen, S., Fang, L., 2023. Diting: A large-scale chinese seismic benchmark dataset for artificial intelligence in seismology. Earthquake Science 36, 84–94.

Ye, L., 2023. Research and design of earthquake detection and phase picking based on transfer learning and attention mechanisms. master’s thesis. Lanzhou University.

Zhu, W., Beroza, G.C., 2019. Phasenet: a deep-neural-network-based seismic arrival-time picking method. Geophysical Journal International 216, 261–273.

Mousavi, S.M., Ellsworth, W.L., Zhu, W., Chuang, L.Y., Beroza, G.C., 2020. Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature communications 11, 3952.
