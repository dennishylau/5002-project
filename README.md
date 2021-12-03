# MSBD-5002-project

## Prerequisite

- conda, either through miniconda or miniforge (osx-arm64)
- Git LFS  
    MacOS: `brew install git-lfs && git lfs install`  
    Others: [See here](https://git-lfs.github.com)

## Getting Started

1. Note: the following files inside data-sets/KPI are too large to be committed, please place them in manually.

   - test-data.hdf
   - train-data.csv

2. `cd` into project directory, and create conda env

    ```sh
    conda env create -f environment.yml
    # as specified in the yml, new 
    # env will be named `5002-project`
    conda activate 5002-project
    ```

    Note: should you encounter any issues, you can also try installing dependencies with pip

    ```sh
    pip install -r requirements.txt
    ```

3. Confirm python version is correct

    ```Python
    python --version
    # Python 3.9.x
    ```

4. cd into `src/` folder
5. Run the main application. A progress bar should show up, and interactive graphs (.html) should start appearing in the `/{projectdir}/output/` folder.

    ```Python
    python main.py
    #  0%|         | 0/250 [00:00<?, ?it/s]
    ```

6. Upon completion, a `submission.csv` file shall appear in the `/{projectdir}/output/` folder.

## Managing Conda + Pip Environment

- Create conda env: `conda env create -f environment.yml`  
- Activate: `conda activate 5002-project`  
- Install pip packages: `pip install -r requirements.txt`
- Export conda package list: `conda env export --no-builds --from-history > environment.yml`  
- Export pip package list: `pip list --format=freeze > requirements.txt`  

## Reference

- [Matrix Profiles Time Series Mining](https://towardsdatascience.com/introduction-to-matrix-profiles-5568f3375d90)
