# vae-ssm

## How to run the code

### Dependencies

- [Python and Conda](https://www.anaconda.com/)
- Setup the conda environment `vae-ssm` by running:

    ```bash
    bash setup_dependencies.sh
    ```

- Don't forget to activate the environment and cd into the src directory when playing with the code later on

    ```bash
    conda activate vae-ssm
    cd src
    ```

### Datasets
- To generate the FRED data, run

    ``` bash
    python gen_fred_dataset.py
    ```
