# few-shot-learning-ts

## How to run the code

### Dependencies

- [Python and Conda](https://www.anaconda.com/)
- Setup the conda environment `fsl-ts` by running:

    ```bash
    bash setup_dependencies.sh
    ```

- Don't forget to activate the environment and cd into the codebase directory when playing with the code later on

    ```bash
    source activate fsl-ts
    cd src
    ```

### Datasets
- To generate the FRED data, run

    ``` bash
    python gen_fred_dataset.py
    ```