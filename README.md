# MSBD-5002-project

## Getting Started

Note: the following files inside data-sets/KPI are too large to be committed, please place them in manually.

- test-data.hdf
- train-data.csv

WIP

## Managing Conda + Pip Environment

- Create conda env: `conda env create -f environment.yml`  
- Activate: `conda activate 5002-project`  
- Install pip packages: `pip install -r requirements.txt`
- Export conda package list: `conda env export --no-builds --from-history > environment.yml`  
- Export pip package list: `pip list --format=freeze > requirements.txt`  

## Reference

- [Matrix Profiles Time Series Mining](https://towardsdatascience.com/introduction-to-matrix-profiles-5568f3375d90)
