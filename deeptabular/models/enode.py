from ..base_models.enode import ENODE
from ..configs.enode_config import DefaultENODEConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class ENODERegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) Regressor. Slightly different with a MLP as a tabular task specific head. This class extends the SklearnBaseRegressor class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from mambular.models import ENODERegressor
        >>> model = ENODERegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ENODE, config=DefaultENODEConfig, **kwargs)


class ENODEClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) Classifier. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseClassifier class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from mambular.models import ENODEClassifier
        >>> model = ENODEClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ENODE, config=DefaultENODEConfig, **kwargs)


class ENODELSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultENODEConfig,
        model_description="""
        Neural Oblivious Decision Ensemble (ENODE) for distributional regression. Slightly different with a MLP as a tabular task specific head.
        This class extends the SklearnBaseLSS class and uses the ENODE model
        with the default ENODE configuration.
        """,
        examples="""
        >>> from mambular.models import ENODELSS
        >>> model = ENODELSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ENODE, config=DefaultENODEConfig, **kwargs)
