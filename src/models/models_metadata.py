from models.MLP import *
from models.RandomForest import *
from models.VRNN import *


models_metadata = {
    
    "rf": RandomForestWrapper,
    "simple_mlp":  NN3Wrapper,
    "vrnn": VRNN,
                   }