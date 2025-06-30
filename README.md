# A Neural Approach for Incomplete Argumentation Frameworks
This project aims to generate neural models capable of determining the acceptability of arguments within incomplete argumentation frameworks. These models are first trained and tested on a dataset generated as part of this project, and then evaluated on the dataset provided by ICCMA 2017.

## Installation
1. Install the **taeydennae** solver from this project : [https://bitbucket.org/andreasniskanen/taeydennae/src/master/](https://bitbucket.org/andreasniskanen/taeydennae/src/master/)
2. Install the **af_reader_py** module from this repository : [https://github.com/Paulo-21/AF-GCN-GAT_wGS](https://github.com/Paulo-21/AF-GCN-GAT_wGS)
3. Install the dependencies listed in the three requirements.txt files located in the Data, GNN, and BigData directories :
```bash
pip install -r Data/requirements.txt
pip install -r GNN/requirements.txt
pip install -r BigData/requirements.txt
```
4. Download the AFs from ICCMA 2017 and transform them into IAFs using the `download_and_run_iaf_generator.sh` script from this repository : [https://bitbucket.org/andreasniskanen/taeydennae/src/master/benchmarks/](https://bitbucket.org/andreasniskanen/taeydennae/src/master/benchmarks/)

## Utilisation
This project provides three main functions, organized into three directories.

### Data : Small dataset
To generate small graphs and label the arguments they contain :
```bash
python3 -m Data.Generation
python3 -m Data.Labeling
```

### GNN : Graph neural networks
To create and train a neural model on the previously generated small dataset :
```bash
python3 -m GNN.Training
```
To predict the acceptability of an argument from the small dataset with a previously trained model :
```bash
python3 -m GNN.iaf_egat_f23_f1 filepath problem-sem argument
```
- **filepath** : path to the `.apx` file (including the `.apx` extension)
- **problem** : decision problem (PCA, NCA, PSA, or NSA)
- **sem** : semantics (ST, PR, or GR)
- **argument** : index of the argument to evaluate

To test the speed and performance of a model on the small dataset:
```bash
python3 -m GNN.Test
```

### BigData : ICCMA 2017 dataset
To predict the acceptability of an argument from the ICCMA 2017 dataset with a previously trained model :
```bash
python3 -m BigData.Big_iaf_egat_f23_f1 filepath problem-sem argument
```
To test the speed and performance of a model on the ICCMA 2017 dataset:
```bash
python3 -m BigData.Test
```

## Configuration
Expliquer le décommantage des lignes pour IAF_root ou sem


## References
