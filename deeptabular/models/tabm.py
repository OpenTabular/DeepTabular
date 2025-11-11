from ..base_models.tabm import TabM
from ..configs.tabm_config import DefaultTabMConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class TabMRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultTabMConfig,
        model_description="""
        TabM regressor. This class extends the SklearnBaseRegressor class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from mambular.models import TabMRegressor
        >>> model = TabMRegressor(ensemble_size=32, model_type='full')
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabM, config=DefaultTabMConfig, **kwargs)


class TabMClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultTabMConfig,
        model_description="""
        TabM classifier. This class extends the SklearnBaseClassifier class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from mambular.models import TabMClassifier
        >>> model = TabMClassifier(ensemble_size=32, model_type='full')
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabM, config=DefaultTabMConfig, **kwargs)


class TabMLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultTabMConfig,
        model_description="""
        TabM for distributional regressoion. This class extends the SklearnBaseLSS class and uses the TabM model
        with the default TabM configuration.
        """,
        examples="""
        >>> from mambular.models import TabMLSS
        >>> model = TabMLSS(ensemble_size=32, model_type='full')
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=TabM, config=DefaultTabMConfig, **kwargs)
