# PHYS490 Research Project
Research project for PHYS 490 course at University of Waterloo, Winter 2020

Paper: https://academic.oup.com/mnras/article/475/3/2978/4775133
Data : https://www.canfar.net/storage/list/starnet/public

---
Authors:

- Names: Matthew Pereira Wilson, Shoshannah Byrne-Mamahit, Shreyas Shankar
- Student IDs: 20644035, 20615468, 20602181

## Dependencies

- json
- time
- numpy
- scipy
- torch
- argparse
- matplotlib
- collections


## Running `main.py`

Implementation done in Python 3.7. To run the script, need to specify ,`--data_path`, the path to the data directory, `--train` and `--test`, which datasets the model will train and test on respectively. Inputs for `--train` and `--test` are either 'real' or 'synthetic'.

```
python3 main.py --data_path=data --train=real --test=real -o=results -c=1
```
