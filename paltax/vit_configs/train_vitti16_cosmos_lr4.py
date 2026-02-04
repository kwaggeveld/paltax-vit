import pathlib
from paltax.TrainConfigs import train_config_npe_base
from ml_collections.config_dict import FieldReference

def get_config():
    """Get the hyperparameter configuration"""
    config = train_config_npe_base.get_config()

    # Overwrite input configuration to created cosmos catalogue
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/input/input_cosmos_train.py'

    config.steps_per_epoch = FieldReference(15600)

    config.model = 'ViT_Ti16'
    config.learning_rate = 1e-4

    return config
