from typing import Tuple
from torch import Tensor

from torch import nn


class DiagonalGaussian(nn.Module):
    """!
    Class implementing a network that maps a vector of size n into two vectors representing the mean
    and variance of a Gaussian with diagonal covariance matrix.
    """

    def __init__(self, input_size : int, nb_components : int) -> None:
        """!
        Constructor.
        @param input_size: size of the vector send as input of the layer
        @param nb_components: the number of components of the diagonal Gaussian
        """
        super().__init__()

        ## @var mean
        # Layer that outputs the mean of the diagonal Gaussian distribution.
        self.mean = nn.Sequential(
            nn.Linear(input_size, nb_components)
        )

        ## @var log_var
        # Layer that outputs the log-variance of the diagonal Gaussian distribution.
        self.log_var = nn.Sequential(
            nn.Linear(input_size, nb_components),
        )

    def forward(self, x : Tensor) -> Tuple[Tensor, Tensor]:
        """!
        Compute the mean and the variance of the diagonal Gaussian.
        @param x: the input vector
        @return the mean and the log of the variance of the diagonal Gaussian
        """
        return self.mean(x), self.log_var(x)
