from ..base_models.saint import SAINT
from ..configs.saint_config import DefaultSAINTConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class SAINTRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultSAINTConfig,
        model_description="""
        SAINT regressor. This class extends the SklearnBaseRegressor
        class and uses the SAINT model with the default SAINT
        configuration.
        """,
        examples="""
        >>> from mambular.models import SAINTRegressor
        >>> model = SAINTRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=SAINT, config=DefaultSAINTConfig, **kwargs)


class SAINTClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultSAINTConfig,
        """SAINT Classifier. This class extends the SklearnBaseClassifier class
        and uses the SAINT model with the default SAINT configuration.""",
        examples="""
        >>> from mambular.models import SAINTClassifier
        >>> model = SAINTClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=SAINT, config=DefaultSAINTConfig, **kwargs)


class SAINTLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultSAINTConfig,
        """SAINT for distributional regression.
        This class extends the SklearnBaseLSS class and uses the
        SAINT model with the default SAINT configuration.""",
        examples="""
        >>> from mambular.models import SAINTLSS
        >>> model = SAINTLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family="normal")
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=SAINT, config=DefaultSAINTConfig, **kwargs)
