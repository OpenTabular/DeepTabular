from ..base_models.tabularnn import TabulaRNN
from ..configs.tabularnn_config import DefaultTabulaRNNConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class TabulaRNNRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultTabulaRNNConfig,
        model_description="""
        TabulaRNN regressor. This class extends the SklearnBaseRegressor
        class and uses the TabulaRNN model with the default TabulaRNN
        configuration.
        """,
        examples="""
        >>> from mambular.models import TabulaRNNRegressor
        >>> model = TabulaRNNRegressor(d_model=64)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabulaRNN, config=DefaultTabulaRNNConfig, **kwargs)


class TabulaRNNClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultTabulaRNNConfig,
        model_description="""
        TabulaRNN classifier. This class extends the SklearnBaseClassifier
        class and uses the TabulaRNN model with the default TabulaRNN
        configuration.
        """,
        examples="""
        >>> from mambular.models import TabulaRNNClassifier
        >>> model = TabulaRNNClassifier(d_model=64)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabulaRNN, config=DefaultTabulaRNNConfig, **kwargs)


class TabulaRNNLSS(SklearnBaseLSS):
    """RNN LSS. This class extends the SklearnBaseLSS class and uses the TabulaRNN model with the default TabulaRNN
    configuration.

    The accepted arguments to the TabulaRNNLSS class include both the attributes in the DefaultTabulaRNNConfig dataclass
    and the parameters for the Preprocessor class.

    Parameters
    ----------
    lr : float, default=1e-04
        Learning rate for the optimizer.
    model_type : str, default="RNN"
        type of model, one of "RNN", "LSTM", "GRU"
    family : str, default=None
        Distributional family to be used for the model.
    lr_patience : int, default=10
        Number of epochs with no improvement after which learning rate will be reduced.
    weight_decay : float, default=1e-06
        Weight decay (L2 penalty) for the optimizer.
    lr_factor : float, default=0.1
        Factor by which the learning rate will be reduced.
    d_model : int, default=64
        Dimensionality of the model.
    n_layers : int, default=8
        Number of layers in the transformer.
    norm : str, default="RMSNorm"
        Normalization method to be used.
    activation : callable, default=nn.SELU()
        Activation function for the transformer.
    embedding_activation : callable, default=nn.Identity()
        Activation function for numerical embeddings.
    head_layer_sizes : list, default=(128, 64, 32)
        Sizes of the layers in the head of the model.
    head_dropout : float, default=0.5
        Dropout rate for the head layers.
    head_skip_layers : bool, default=False
        Whether to skip layers in the head.
    head_activation : callable, default=nn.SELU()
        Activation function for the head layers.
    head_use_batch_norm : bool, default=False
        Whether to use batch normalization in the head layers.
    layer_norm_after_embedding : bool, default=False
        Whether to apply layer normalization after embedding.
    pooling_method : str, default="cls"
        Pooling method to be used ('cls', 'avg', etc.).
    norm_first : bool, default=False
        Whether to apply normalization before other operations in each transformer block.
    bias : bool, default=True
        Whether to use bias in the linear layers.
    rnn_activation : callable, default=nn.SELU()
        Activation function for the transformer layers.
    bidirectional : bool, default=False.
        Whether to process data bidirectionally
    cat_encoding : str, default="int"
        Encoding method for categorical features.
    n_bins : int, default=50
        The number of bins to use for numerical feature binning. This parameter is relevant
        only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    numerical_preprocessing : str, default="ple"
        The preprocessing strategy for numerical features. Valid options are
        'binning', 'one_hot', 'standardization', and 'normalization'.
    use_decision_tree_bins : bool, default=False
        If True, uses decision tree regression/classification to determine
        optimal bin edges for numerical feature binning. This parameter is
        relevant only if `numerical_preprocessing` is set to 'binning' or 'one_hot'.
    binning_strategy : str, default="uniform"
        Defines the strategy for binning numerical features. Options include 'uniform',
        'quantile', or other sklearn-compatible strategies.
    cat_cutoff : float or int, default=0.03
        Indicates the cutoff after which integer values are treated as categorical.
        If float, it's treated as a percentage. If int, it's the maximum number of
        unique values for a column to be considered categorical.
    treat_all_integers_as_numerical : bool, default=False
        If True, all integer columns will be treated as numerical, regardless
        of their unique value count or proportion.
    degree : int, default=3
        The degree of the polynomial features to be used in preprocessing.
    knots : int, default=12
        The number of knots to be used in spline transformations.
    """

    def __init__(self, **kwargs):
        super().__init__(model=TabulaRNN, config=DefaultTabulaRNNConfig, **kwargs)
