from abc import ABC, abstractmethod


class AbstractTest(ABC):
    """
    An abstract base class for statistical tests.
    This class defines the interface for statistical tests, requiring implementations
    to provide methods for calculating results from a Z-score, parameters, and data.
    """

    @abstractmethod
    def from_score(self):
        """Calculate results from a Z-score or T-score."""
        pass

    @abstractmethod
    def from_parameters(self):
        """Calculate results from parameters."""
        pass

    @abstractmethod
    def from_data(self):
        """Calculate results from data."""
        pass
