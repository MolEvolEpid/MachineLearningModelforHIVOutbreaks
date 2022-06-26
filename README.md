# Machine Learning Models for HIV Outbreaks

Code from _Kupperman et al._ "A deep learning approach to real-time HIV outbreak detection using genetic data".

This repository is organized into two directories.

* The `Python` module contains all Python codes for handling the deep learning and data analysis.
* The `HIV_simulator` module contains all R codes to simulate an HIV outbreak.

## How to use this repository to analyze your own data

### 1. Create a new [Anaconda environment](https://www.anaconda.com/products/individual) (if you do not have one already).

```bash
conda create -n MLforHIV python=3.9 numpy scipy matplotlib scikit-learn jupyter pandas
```
`Pandas` is required for interfacing with HIV-Trace.

Install R dependencies.

```R
install.packages(c("here", "R.matlab", "abind", "tictoc", "iterators", "foreach", "seriation", "doParallel"))
```
The package `parallel` is also required but is traditionally bundled with the base environment. 

### 2. Install tensorflow

The CPU version will suffice, but GPU acceleration is useful for many use cases. 
The CPU version can be installed with `pip install tensorflow`. 
[GPU acceleartion requires additional dependencies](https://www.tensorflow.org/install) and is limited to Windows and Ubuntu derivatives.

### 3. Clone this repository onto your machine

Using the git command line API, download this git repository.

```bash
git clone git@github.com:MolEvolEpid/MachineLearningModelforHIVOutbreaks.git
```

### 4. Generate training data

A reference test set is provided if you wish to use your own simulator data. Documentation for the data output 
format is described [here](/HIV_simulator/ReadMe.md).
To use our code, navigate to `/HIV_simulator` and execute the provided R script.

```bash
Rscript RunMe_15.R
```
The outputs of this script are time-stamped so many different sets can be created. 

### 5. Train models using the training data

We provide a CLI interface to our training script. On some operating systems, you may need to mark the script as executable.
On linux:
```
chmod +x Python/MakeAndTrain_CLI.py
```

The general syntax is

```bash
python Python/MakeAndTrain_CLI.py --test /path/to/data.mat --text [paths/to/data.mat] --ordering "None"
```

You can rename the files, or use the following bash command.

```bash
python Python/MakeAndTrain_CLI.py --test $(find -name *TEST-*) --train $(find -name *TRAIN-*) --ordering "None"
```
To use a second keyword in the file path,
```bash
SIZE=15
python Python/MakeAndTrain_CLI.py --test $(find -name W*$SIZE*TEST-*) --train $(find -name W*$SIZE *TRAIN-*) --ordering "None"
```

To obtain full use of the repository or interface with a SLURM scheduler, a sample batch script is provided below

```bash

#!/bin/bash
# Provide #SBATCH directives here
KEY="Uniform_exits_mu67_synth"  # adjust me to the directory name for your saved data
CORE="*Example_Data/${KEY}" 

for SIZE in "15" "20" "30" "40" "50"  # adjust me to the sizes in your generated data
do
    TEST="${CORE}*data_${SIZE}*TEST"  # specify test alias
    TRAIN="${CORE}*data_${SIZE}*TRAIN"  # specify training alias
    # for debugging, log the search path/resolution to console
    echo $TEST  
    echo "${TEST}*W${SIZE}-*.mat" 
    echo $(find -wholename "${TEST}*W${SIZE}*.mat")  # Debugging: echo the test set we found
    python3 ./MakeAndTrain_CLI.py --test $(find '../' -wholename "${TEST}*W${SIZE}*.mat")  --train $(find '../' -wholename "${TRAIN}*W${SIZE}*.mat") --savekey $SKEY --ordering "None"
    echo "Script called and finished"
done
source deactivate  # for SLURM - you may not need to deactivate your environment
echo "Conda Env deactivated. Job exiting."
```

### 6. Convert your data into a sequence of matrices

1. Sort the collection of sequences from the oldest sequence (at the beginning) to youngest sequence (at the end).
   Perform a pairwise alignment on all sequences using your favorite multiple sequence alignment algorithm. Record
   the length of the aligned sequence. We will need this to convert from an evolutionary distance (substitutions per
   site) to an estimated number of substitutions.
2. Using your favorite DNA evolutionary distance calculation tool/package, compute the pairwise distance between the first
   two sequences in the list. Convert this to the matrix representation if this is not the default output format for
   your DNA evolutionary distance tool.
3. Save the matrix and list of labels (that match the rows/columns of the matrix). Store the matrix in `matrices`
   and the labels in `row_labels`.
   1. The saved **file names** should be saved in a single (otherwise empty) directory and contain a single "number-like" term separated by a `-` ignoring the file type `.mat`. This is necessary to recover the correct order of files. **Good** file names are `EvoDist-01.mat` or `Example-23-series.mat`. Avoid using a sample date or any other number-like terms in the file name. The first `-`-delimited character sting that can be converted to a valid integer is accepted as the index. If no index is obtained, an error will be raised in the Python code.
   2. Data files should be saved as a `.mat` Matlab V7.0 file. If you are in `R`, we recommend the `R.matlab`
   package. In Python, the `scipy.io` module provides this functionality.

4. Return to step 2 and repeat with one additional sequence until you have reached the most recent sample.

### 7. Use the trained model to analyze the data

We provide a pretrained model, the data analyzed, and the code necessary to regenerate the plot in figure 7 in [this
notebook](/Notebooks/TimeSeries%20Presentation.ipynb). Generating data and training the model is still
computationally expensive, _we recommend using the pre-trained models for most users_.

If you are using a newly trained model, update the directory path with the trained model in the third cell.

If you are using your own data, update the directory path and `scale_factor` in the fourth cell. `scale_factor` is 
the value used to rescale evolutionary distances into the estimated number of mutations. These distances are **not** 
rounded to the nearest integer. For most users, 
the `scale_factor` is the length of the aligned sequence, or the length of the reference 
used to compute the alignment in part 6 above.
