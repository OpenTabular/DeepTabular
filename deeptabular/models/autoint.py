from ..base_models.autoint import AutoInt
from ..configs.autoint_config import DefaultAutoIntConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class AutoIntRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultAutoIntConfig,
        model_description="""
        AutoInt regressor. This class extends the SklearnBaseRegressor
        class and uses the AutoInt model with the default AutoInt
        configuration.
        """,
        examples="""
        >>> from mambular.models import AutoIntRegressor
        >>> model = AutoIntRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=AutoInt, config=DefaultAutoIntConfig, **kwargs)


class AutoIntClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultAutoIntConfig,
        """AutoInt Classifier. This class extends the SklearnBaseClassifier class
        and uses the AutoInt model with the default AutoInt configuration.""",
        examples="""
        >>> from mambular.models import AutoIntClassifier
        >>> model = AutoIntClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=AutoInt, config=DefaultAutoIntConfig, **kwargs)


class AutoIntLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultAutoIntConfig,
        """AutoInt for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        AutoInt model with the default AutoInt configuration.""",
        examples="""
        >>> from mambular.models import AutoIntLSS
        >>> model = AutoIntLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family="normal")
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=AutoInt, config=DefaultAutoIntConfig, **kwargs)
