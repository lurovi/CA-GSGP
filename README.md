# CA-GSGP 

Environment with Python 3.10.11.

Main libraries:

genepro (version 1.2.0) -> https://github.com/giorgia-nadizar/genepro/

You can find the requirements.txt in the project folder.

Be careful to properly set PYTHONPATH environment variable. Plus, mind that code leverages a custom environment variable called CURRENT\_CODEBASE\_FOLDER that contains the absolute path of the folder (the path ends with '/') that contains all code and data relevant for the project. You just have to set this variable properly to make it easier to access locations for code and data. Change the paths in the experiment scripts accordingly, always be careful that for paths pointing to a folder you must always put a '/' at the end.

For the datasets, the scripts for processing them can be found in the benchmark package. The experiments expect datasets in .csv format named trainSEED.csv and testSEED.csv where SEED is the integer value of the seed. Each dataset has d + 1 columns where d is the number of variables/features. The column names for the variables range from 0 to d-1. The last column is called target (the y values). Column names are available in the header of the .csv file (very first row of the file). Besides the header, each row is a dataset observation/record. All train-test splits belonging to the same dataset go into the same directory called with the name of the dataset (e.g., vladislavleva14/, airfoil/).

The main scripts for experiments can be found in the exps package.
