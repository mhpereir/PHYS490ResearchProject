# PHYS490 Research Project
Final research project for the course PHYS 490 - Machine Learning in Physical Sciences at the University of Waterloo, Winter 2020

---
Authors:

- Names: Matthew Pereira Wilson, Shoshannah Byrne-Mamahit, Shreyas Shankar
- Student IDs: 20644035, 20615468, 20602181

**StarNet** is a convolutional neural network model that predicts stellar parameters from stellar spectra, given a set of previously estimated stellar parameters. This program is designed to work with the datasets described in the paper *An application of deep learning in the analysis of stellar spectra, Fabbro et al (2017)*. The pre-processed datasets used are `ASSET.h5`, `test_data.h5`, and `training_data.h5`, all of which can be found at the URL below within the base directory `/starnet/public`.

Paper: https://academic.oup.com/mnras/article/475/3/2978/4775133<br/>
Data : https://www.canfar.net/storage/list/starnet/public

## Dependencies

- python3.x
- json
- argparse
- time
- os
- h5py
- numpy
- tqdm
- matplotlib
- scipy
- torch (PyTorch v1.4.0)

## Running `main.py`

Assuming that all the dependencies and supporting files are cloned from this repo, `main.py` can be run from within this repo with the following example command:
```
python3 main.py -d ./data --train real --test real -o ./results -v True -c True -s True -m True
```
Let's break down the anatomy of this command:
* `-d, --data_path`: This required argument points the program to the directory path where all the training and testing datasets are stored, as well as the necessary hyperparameters file (elaborated below).
* `--train`: This flag specifies whether to train the model on *real* data or *synthetic* data. The argument parser will not accept values other than {'real', 'synthetic'}.
* `--test`: This flag specifies whether to test the model on *real* data or *synthetic* data. The argument parser will not accept values other than {'real', 'synthetic'}.
* `-o, --output_path`: This optional flag points the program to the directory path in which to store any generated products from the program, such as loss & residual plots, and the saved model. The program will create a folder with the provided name if the folder does not already exist. By default the output directory is `./results`.
* `-v, --verbose`: This optional boolean flag toggles the verbosity of the program. If enabled (bool: True), the program will print update messages to stdout at the verbosity interval specified in the hyperparameters file (elaborated below). By default this flag is set to False.
* `-c, --cuda`: This optional boolean flag toggles the usage of CUDA GPU resources. If enabled (bool: True), the program will try to load the model parameters and tensors onto the GPU's VRAM, allowing for considerable performance speed-ups. While the dataset will only be loaded into the system's DRAM, due to the size of the tensors computed by the model it is highly recommended that this flag be enabled ***only*** if your GPU has at least 3 GB of VRAM. By default it is disabled (set to False). If enabled and the GPU cannot allocate enough memory for computation, the program will terminate; the program is not designed to fall back to CPU computation if GPU fails.
* `-s, --save`: This optional boolean flag toggles saving of the trained model's learned state parameters, for loading and inference at a later time. It will always save to the directory specified in `--output_path`. It is enabled (bool: True) by default.
* `-m, --max_cpu`: This optional flag toggles maximum CPU usage. If a CUDA GPU is not available or `--cuda` is disabled, enabling this flag (bool: True) will allow the program to train the model using the full capacity of all the cores available on the machine. Doing so will restrict other system processes from functioning properly, and on a laptop running several background applications this could potentially lead to decrease in training speed as processes wrestle for compute resources. It is recommended that this flag only be run on servers with ample compute resources and minimal background processes. Additionally, even if this flag is enabled the program will first ensure that the machine has sufficient compute resources to make use of this flag; if not the program will use only half the capacity of the CPU cores. By default this flag is disabled (set to False).

Note that all of the boolean flags are case-insensitive and configured to work with any of the following input strings: 'yes', 'y', 'true', 't', '1', 'no', 'none', 'n', 'false', 'f', '0'.

Consider the following case: data files stores in `../data_files`, train on synthetic data, test on real data, default save path, non-verbose, cpu-only, not maxed out, and save model. The command to enable this would be:
```
python3 main.py -d ../data_files --train synthetic --test real
```

For quick reference to an abridged form of this help guide, enter the command:
```
python3 main.py -h
```

## Hyperparameter JSON FILE

The hyperparameter file provided, `params.json`, is located in [data](./data). It is already tuned for performance with the model. Any user-provided hyperparameter file must contain the following parameters:
* `n_epoch` : The maximum number of epochs that the model is allowed to train for. Note that it may finish training earlier than this due to early stopping.
* `n_cross` : The number of spectra used for computing cross-validation loss.
* `lr` : The learning rate of the network's optimizer.
* `n_epoch_v` : The epoch interval for which the program will print regular progress updates to stdout during training (only needed if verbose flag is not False)
* `n_mini_batch` : The batch size of data used for each training iteration
* `n_rank`: To prevent the program from overflowing the machine's DRAM, `n_rank` can be set to a value greater than 1 to split the dataset into n_rank number of "batches", which are loaded into memory as needed. 7214/n_rank is the size of an individual batch. Set to 1 to load the entire dataset into RAM.
