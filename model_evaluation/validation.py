"""
    A program to evaluate trained Paltax neural network models on a fixed
    validation dataset and record loss metrics as a function of training
    checkpoint.

    Usage:
        python validation.py \
            --workdir=/path/to/training_run_directory \
            --config=/path/to/config.py

    Inputs:
        - workdir: Directory containing model checkpoints produced during
          training (named checkpoint_<step>).
        - config:  Training configuration file used to define the model,
          same as used during training.

    Outputs:
        val-<run_name>.npz:
            NumPy archive containing:
                - run_name:        name of the training run
                - model:           model architecture used
                - num_params:      total number of trainable parameters
                - steps:           checkpoint training steps
                - losses:          Gaussian loss per checkpoint
                - losses_ss:       loss on sigma_sub per checkpoint
                - num_val_images:  number of validation images evaluated
                - image_size:      input image size
                - batch_size:      training batch size from config

    Notes:
        - Assumes validation images are stored in ./validation_images, 
          as .npz files. Each file must contain:
            * "images": array of shape (N, H, W)
            * "truths": array of shape (N, D)
        - Learning rate schedules and model definitions are reconstructed
          from the provided training configuration.

    Written by Koen Waggeveld, k.c.waggeveld@student.rug.nl.
"""


import jax
import jax.numpy as jnp

from paltax import train, models
from paltax.train import gaussian_loss, sigma_sub_loss, create_train_state, get_learning_rate_schedule
from flax.training import checkpoints

from tqdm import tqdm

from absl import app, flags
from ml_collections import config_flags
from pathlib import Path
import numpy as np

# ============================================================
#  Global variables
# ============================================================

RNG = None
CONFIG = None
MODEL = None
IMAGE_SIZE = None
LEARNING_RATE_SCHEDULE = None

# ============================================================
#  Input parsing
# ============================================================

FLAGS = flags.FLAGS
flags.DEFINE_string('workdir', None, 'working directory.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training configuration.',
)

# ============================================================
#  Functions
# ============================================================

def clean_train_state():
    return create_train_state(
            RNG,
            CONFIG,
            MODEL,
            IMAGE_SIZE,
            LEARNING_RATE_SCHEDULE
        )

def load_validation_data():
    images = []
    truths = []

    for image_batch in Path("./validation_images").iterdir():
        batch = np.load(image_batch, allow_pickle = True)
        images.append(batch["images"])
        truths.append(batch["truths"])

    images = np.concatenate(images, axis = 0)   # (1024, H, W)
    truths = np.concatenate(truths, axis = 0)   # (1024, D)

    images = jnp.expand_dims(jnp.asarray(images), axis = -1)  # (1024, H, W, 1)
    truths = jnp.asarray(truths)

    return images, truths

@jax.jit
def evaluate_losses(params, batch_stats, images, truths):
    outputs = MODEL.apply(
        {'params': params, 'batch_stats': batch_stats},
        images, train = False, mutable = False,
    )

    n = len(outputs)
    
    return (
        gaussian_loss(outputs, truths) / n,
        sigma_sub_loss(outputs, truths) / n,
    )

def set_global_variables():
    global RNG, CONFIG, MODEL, IMAGE_SIZE, LEARNING_RATE_SCHEDULE

    CONFIG = FLAGS.config

    input_config = train._get_config(CONFIG.input_config_path)

    # Initialise model
    model_cls = getattr(models, CONFIG.model)
    MODEL = model_cls(
        num_outputs = len(input_config['truth_parameters'][0]) * 2,
        dtype = jnp.float32
    )

    IMAGE_SIZE = input_config['kwargs_detector']['n_x']

    LEARNING_RATE_SCHEDULE = get_learning_rate_schedule(
        CONFIG, CONFIG.learning_rate * CONFIG.batch_size / 256.0
    )

    RNG = jax.random.PRNGKey(CONFIG.get('rng_key', 0))
    return

# ============================================================
#  Main body
# ============================================================

def main(_):
    set_global_variables()

    workdir = FLAGS.workdir

    images, truths = load_validation_data()

    losses = []
    losses_ss = []

    checkpoint_list = sorted(
        [p for p in Path(workdir).iterdir() if p.is_dir() and p.name.startswith("checkpoint_")],
        key = lambda p: int(p.name.split("_")[1])
    )
    
    state = None
    for checkpoint in tqdm(checkpoint_list):
        state = clean_train_state()
        state = checkpoints.restore_checkpoint(checkpoint, state)


        loss, loss_ss = evaluate_losses(
            state.params, state.batch_stats, images, truths
        )

        losses.append(loss)
        losses_ss.append(loss_ss)

    steps = np.array([int(p.name.split("_")[1]) for p in checkpoint_list])

    np.savez(
        f"val-{Path(workdir).name}.npz",

        # Model & run info
        run_name   = Path(workdir).name,
        model      = CONFIG.model,
        num_params = sum(x.size for x in jax.tree_util.tree_leaves(state.params)),

        # x-axis information
        steps      = np.asarray(steps, dtype=np.int64),

        # Losses
        losses     = np.asarray(losses, dtype=np.float32),
        losses_ss  = np.asarray(losses_ss, dtype=np.float32),

        # Evaluation info
        num_val_images = images.shape[0],
        image_size     = IMAGE_SIZE,
        batch_size     = CONFIG.batch_size,
    )


if __name__ == '__main__':
    app.run(main)