from input_config_br_modified import get_config as input_config()
from paltax import source_models
from pathlib import Path

def get_config():
    config = input_config()

    cosmos_path = str(Path(__file__).parent.parent.parent.parent)
    cosmos_path += '/datasets/cosmos/COSMOS_train.h5'

    config['all_models']['all_source_models']= (source_models.CosmosCatalog(cosmos_path),)

    return config