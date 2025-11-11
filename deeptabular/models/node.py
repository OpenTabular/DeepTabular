from ..base_models.node import NODE
from ..configs.node_config import DefaultNODEConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class NODERegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultNODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) Regressor. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseRegressor class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from mambular.models import NODERegressor
        >>> model = NODERegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=NODE, config=DefaultNODEConfig, **kwargs)


class NODEClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultNODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) Classifier. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseClassifier class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from mambular.models import NODEClassifier
        >>> model = NODEClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=NODE, config=DefaultNODEConfig, **kwargs)


class NODELSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultNODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (NODE) for distributional regression. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseLSS class and uses the NODE model
        with the default NODE configuration.
        """,
        examples="""
        >>> from mambular.models import NODELSS
        >>> model = NODELSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=NODE, config=DefaultNODEConfig, **kwargs)
