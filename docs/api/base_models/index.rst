.. -*- mode: rst -*-

.. currentmodule:: mambular.base_models

BaseModels
==========

This module provides foundational classes and architectures for Mambular models, including various neural network architectures tailored for tabular data.

=========================================    =======================================================================================================
Modules                                       Description
=========================================    =======================================================================================================
:class:`BaseModel`                            Abstract base class defining the core structure and initialization logic for Mambular models.
:class:`TaskModel`                            PyTorch Lightning module for managing model training, validation, and testing workflows.
:class:`Mambular`                             Flexible neural network model leveraging the Mamba architecture with configurable normalization techniques for tabular data.
:class:`MLP`                                  Multi-layer perceptron (MLP) model designed for tabular tasks, initialized with a custom configuration.
:class:`ResNet`                               Deep residual network (ResNet) model optimized for structured/tabular datasets.
:class:`FTTransformer`                        Feature Tokenizer (FTTransformer) model for tabular tasks, incorporating advanced embedding and normalization techniques.
:class:`TabTransformer`                       TabTransformer model leveraging attention mechanisms for tabular data processing.
:class:`NODE`                                 Neural Oblivious Decision Ensembles (NODE) for tabular tasks, combining decision tree logic with deep learning.
:class:`TabM`                                 TabM architecture designed for tabular data, implementing batch-ensembling MLP techniques.
:class:`NDTF`                                 Neural Decision Tree Forest (NDTF) model for tabular tasks, blending decision tree concepts with neural networks.
:class:`TabulaRNN`                            Recurrent neural network (RNN) model, including LSTM and GRU architectures, tailored for sequential or time-series tabular data.
:class:`MambAttention`                        Attention-based architecture for tabular tasks, combining feature importance weighting with advanced normalization techniques.
:class:`SAINT`                                SAINT model. Transformer based model using row and column attetion.
=========================================    =======================================================================================================


.. toctree::
   :maxdepth: 1

   BaseModels
