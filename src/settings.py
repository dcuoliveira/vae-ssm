import os

# TODO: all devs must set up an env variable with this key
FRED_KEY = "12d77a40907e43a92e9a295801db18d2"

# database reference: https://research.stlouisfed.org/wp/more/2015-012
FRED_MD_PATH = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"

SOURCE_PATH = os.path.dirname(__file__)
INPUTS_PATH = os.path.join(SOURCE_PATH, "data", "inputs")
OUTPUTS_PATH = os.path.join(SOURCE_PATH, "data", "outputs")
DATA_UTILS_PATH = os.path.join(SOURCE_PATH, "data", "utils")