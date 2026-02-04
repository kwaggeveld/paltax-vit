Fork notice
===========

This repository is a fork of ``paltax``.
The original ``paltax`` implementation is preserved.
The additions documented below were introduced for an associated report and
do not alter the original scope of the package beyond what is explicitly
listed.

Additions in this fork
----------------------

The following functionality was added on top of the original ``paltax``
implementation.

Loss function extension
^^^^^^^^^^^^^^^^^^^^^^^

- A ``sigma_sub_loss`` function was added to ``paltax/train.py``.
  This function computes a Gaussian loss on the ``sigma_sub`` parameter.
- The ``compute_metrics()`` function was extended to report the
  corresponding ``sigma_sub_loss`` during training and evaluation.

Vision Transformer models
^^^^^^^^^^^^^^^^^^^^^^^^^

- Several Vision Transformer (ViT) architectures were added to
  ``paltax/models.py``.
- These implementations are derived from the official
  Google Research Vision Transformer repository:
  https://github.com/google-research/vision_transformer
- For convenience, the relevant source files from that repository were copied
  into ``paltax/vit_jax/``:
  
  - ``models_resnet.py``
  - ``models_vit.py``

Model evaluation
^^^^^^^^^^^^^^^^

- An evaluation script was added at ``model_evaluation/validation.py``.
- The script takes a configuration file and working directory, which
  should be identical to those passed to `main.py` for training,
  and evaluates all checkpoints stored in `workdir`.
- Validation is performed on 1024 images stored in
  ``model_evaluation/validation_images/``.
- The script outputs a NumPy archive named ``val-[workdir name].npz``
  containing training and validation information,
  see ``validation.py`` for details.

COSMOS dataset processing
^^^^^^^^^^^^^^^^^^^^^^^^^

- A program was added at ``datasets/cosmos/cosmos_catalogue_cuts.py`` to process the COSMOS real galaxy dataset and generate
  datasets used as light sources in ``paltax``.
- The following datasets are produced:

  - ``COSMOS_train.h5``: training dataset, containing 2,163 images
  - ``COSMOS_test.h5``: validation dataset, containing 99 images

- The aforementioned validation images used by ``validation.py`` were simulated using ``COSMOS_test.h5`` as light sources.

----


==========================================================================
|logo| paltax
==========================================================================

.. |logo| image:: https://raw.githubusercontent.com/swagnercarena/paltax/main/docs/figures/logo.png
    	:target: https://raw.githubusercontent.com/swagnercarena/paltax/main/docs/figures/logo.png
    	:width: 100

.. |ci| image:: https://github.com/swagnercarena/paltax/workflows/CI/badge.svg
    :target: https://github.com/swagnercarena/paltax/actions

.. |coverage| image:: https://coveralls.io/repos/github/swagnercarena/paltax/badge.svg?branch=main
	:target: https://coveralls.io/github/swagnercarena/paltax?branch=main

.. |license| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
    :target: https://github.com/swagnercarena/paltax/main/LICENSE

|ci| |coverage| |license|

``paltax`` is a package for conducting simulation-based inference on strong gravitational lensing images.

Installation
------------

``paltax`` is installable via pip:

.. code-block:: bash

    $ pip install paltax

For the most up-to-date version of paltax install directly from the git repository.

.. code-block:: bash

    $ git clone https://github.com/swagnercarena/paltax.git
	$ cd path/to/paltax/
	$ pip install -e .

Usage
-----

The main functionality of ``paltax`` is to train (sequential) neural posterior estimators with on-the-fly data generation. To train a model with ``paltax`` you need a training configuration file that is passed to main.py:

.. code-block:: bash

    $ python main.py --workdir=path/to/model/output/folder --config=path/to/training/configuration

``paltax`` comes preloaded with a number of training configuration files which are described in ``paltax/TrainConfigs/README.rst``. These training configuration files require input configuration files, examples of which can be found in ``paltax/InputConfigs/``.

Demos
-----

``paltax`` comes with a tutorial notebook for users interested in using the package.

* `Using an input configuration file to generate a batch of images <https://github.com/swagnercarena/paltax/blob/main/notebooks/GenerateImages.ipynb>`_.

Figures
-------

Code for generating the plots included in some of the publications using ``paltax`` can be found under the corresponding arxiv number in the ``paltax/notebooks/papers/`` folder.

Attribution
-----------
If you use ``paltax`` for your own research, please cite the ``paltax`` package (`Wagner-Carena et al. 2024 <https://arxiv.org/abs/2404.14487>`_)

``paltax`` builds off of the publically released Google DeepMind codebase `jaxstronomy <https://github.com/google-research/google-research/tree/master/jaxstronomy>`_.
