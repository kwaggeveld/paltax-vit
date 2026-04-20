import pathlib
from paltax.TrainConfigs import train_config_npe_base
from ml_collections.config_dict import FieldReference, placeholder

'''
    Base configuration for training ViT models in paltax. 
    Uses base NPE configuration with COSMOS input config,
    higher steps_per_epoch and lower learning rate.
    

    Use as:
        python main.py \
            --config=/path/to/train_vit_base.py \
            --config.model=ModelName \
            --config.other_fields=value \
            --workdir=/path/to/workdir

   
'''


def get_config():
    """Get the hyperparameter configuration"""
    config = train_config_npe_base.get_config()

    # Overwrite input configuration to created cosmos catalogue
    config.input_config_path = str(pathlib.Path(__file__).parent)
    config.input_config_path += '/input/input_cosmos_train.py'

    config.steps_per_epoch = FieldReference(15600)
    config.learning_rate = 1e-4

    config.model = placeholder(str)     # model name should be set through the `--config.model=model_name` flag

    config.model_kwargs = {
        "dropout_rate": 5,
        "num_layers": 12
    }

    return config
