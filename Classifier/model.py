from abc import ABC, abstractmethod
from typing import Any
from sklearn.base import ClassifierMixin


class BaseModel(ABC):
    def __init__(self, model, **kwargs: Any) -> None:
        """Initialize the base model with optional parameters."""
        self.params = kwargs
        self.model = model

    @abstractmethod
    def train(self, data) -> None:
        """Train the model with the provided data."""
        self.model.fit(data)

    def predict(self, data):
        """Make predictions using the trained model."""
        return self.model.predict(data) 

    @property
    def name(self) -> str:
        """Return the name of the model."""
        return self.__class__.__name__
    
    def score(self, X, y):
        """Return the name of the model."""
        return self.model.score(X, y)