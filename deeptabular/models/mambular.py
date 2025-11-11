from ..base_models.mambular import Mambular
from ..configs.mambular_config import DefaultMambularConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class MambularRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultMambularConfig,
        model_description="""
        Mambular regressor. This class extends the SklearnBaseRegressor class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from mambular.models import MambularRegressor
        >>> model = MambularRegressor(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)


class MambularClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultMambularConfig,
        model_description="""
        Mambular classifier. This class extends the SklearnBaseClassifier class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from mambular.models import MambularClassifier
        >>> model = MambularClassifier(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)


class MambularLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultMambularConfig,
        model_description="""
        Mambular LSS for distributional regression. This class extends the SklearnBaseLSS class and uses the Mambular model
        with the default Mambular configuration.
        """,
        examples="""
        >>> from mambular.models import MambularLSS
        >>> model = MambularLSS(d_model=64, n_layers=8)
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=Mambular, config=DefaultMambularConfig, **kwargs)
