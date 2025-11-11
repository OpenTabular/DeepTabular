from ..base_models.mlp import MLP
from ..configs.mlp_config import DefaultMLPConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class MLPRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultMLPConfig,
        model_description="""
        Multi-Layer Perceptron regressor. This class extends the SklearnBaseRegressor class and uses the MLP model
        with the default MLP configuration.
        """,
        examples="""
        >>> from mambular.models import MLPRegressor
        >>> model = MLPRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)


class MLPClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultMLPConfig,
        model_description="""
        Multi-Layer Perceptron classifier This class extends the SklearnBaseClassifier class and uses the MLP model
        with the default MLP configuration.
        """,
        examples="""
        >>> from mambular.models import MLPClassifier
        >>> model = MLPClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)


class MLPLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultMLPConfig,
        model_description="""
        Multi-Layer Perceptron for distributional regression. This class extends the SklearnBaseLSS class and uses the MLP model
        with the default MLP configuration.
        """,
        examples="""
        >>> from mambular.models import MLPLSS
        >>> model = MLPLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=MLP, config=DefaultMLPConfig, **kwargs)
