from ..base_models.resnet import ResNet
from ..configs.resnet_config import DefaultResNetConfig
from ..utils.docstring_generator import generate_docstring
from .utils.sklearn_base_classifier import SklearnBaseClassifier
from .utils.sklearn_base_lss import SklearnBaseLSS
from .utils.sklearn_base_regressor import SklearnBaseRegressor


class ResNetRegressor(SklearnBaseRegressor):
    __doc__ = generate_docstring(
        DefaultResNetConfig,
        model_description="""
        ResNet regressor. This class extends the SklearnBaseRegressor class and uses the ResNet model
        with the default ResNet configuration.
        """,
        examples="""
        >>> from mambular.models import ResNetRegressor
        >>> model = ResNetRegressor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)


class ResNetClassifier(SklearnBaseClassifier):
    __doc__ = generate_docstring(
        DefaultResNetConfig,
        model_description="""
        ResNet classifier This class extends the SklearnBaseClassifier class and uses the ResNet model
        with the default ResNet configuration.
        """,
        examples="""
        >>> from mambular.models import ResNetClassifier
        >>> model = ResNetClassifier()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)


class ResNetLSS(SklearnBaseLSS):
    __doc__ = generate_docstring(
        DefaultResNetConfig,
        model_description="""
        ResNet for distributional regressor. This class extends the SklearnBaseLSS class and uses the ResNet model
        with the default ResNet configuration.
        """,
        examples="""
        >>> from mambular.models import ResNetLSS
        >>> model = ResNetLSS()
        >>> model.fit(X_train, y_train, family='normal')
        >>> preds = model.predict(X_test)
        >>> model.evaluate(X_test, y_test)
        """,
    )

    def __init__(self, **kwargs):
        super().__init__(model=ResNet, config=DefaultResNetConfig, **kwargs)
