from typing import Optional

from torch import nn, Tensor
import torch

from relab import relab


class ContinuousDecoderNetwork(nn.Module):
    """!
    @brief A convolutional decoder for 84x84 images with continuous latent variables.
    """

    def __init__(self, n_continuous_vars : int = 10, stack_size : Optional[int] = None) -> None:
        """!
        Constructor.
        @param n_continuous_vars: the number of continuous latent variables
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var lin_net
        # Linear layers that process the continuous latent variables.
        self.lin_net = nn.Sequential(
            nn.Linear(n_continuous_vars, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11 * 11 * 64),
            nn.ReLU()
        )

        ## @var stack_size
        # Number of stacked frames in each observation
        self.stack_size = 1
        # TODO self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var up_conv_net
        # Transposed convolution layers that upscale the feature maps to the final image.
        self.up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.stack_size, (3, 3), stride=(1, 1), padding=(3, 3), output_padding=(0, 0))
        )

    def forward(self, x : Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the reconstructed image
        """
        x = self.lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 11, 11))
        return self.up_conv_net(x)


class DiscreteDecoderNetwork(nn.Module):
    """!
    @brief A convolutional decoder for 84x84 images with discrete latent variables.
    """

    def __init__(
        self,
        n_discrete_vars : int = 20,
        n_discrete_vals : int = 10,
        stack_size : Optional[int] = None
    ) -> None:
        """!
        Constructor.
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var lin_net
        # Linear layers that process the discrete latent variables.
        self.lin_net = nn.Sequential(
            nn.Linear(n_discrete_vars * n_discrete_vals, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11 * 11 * 64),
            nn.ReLU()
        )

        ## @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var up_conv_net
        # Transposed convolution layers that upscale the feature maps to the final image.
        self.up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.stack_size, (3, 3), stride=(1, 1), padding=(3, 3), output_padding=(0, 0))
        )

    def forward(self, x : Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the reconstructed image
        """
        x = self.lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 11, 11))
        return self.up_conv_net(x)


class MixedDecoderNetwork(nn.Module):
    """!
    @brief A convolutional decoder for 84x84 images with discrete and continuous latent variables.
    """

    def __init__(
        self,
        n_continuous_vars : int = 10,
        n_discrete_vars : int = 20,
        n_discrete_vals : int = 10,
        stack_size : Optional[int] = None
    ):
        """!
        Constructor.
        @param n_continuous_vars: the number of continuous latent variables
        @param n_discrete_vars: the number of discrete latent variables
        @param n_discrete_vals: the number of values taken by all the discrete latent variables,
            or a list describing the number of values taken by each discrete latent variable
        @param stack_size: the number of stacked frame in each observation, if None use the configuration
        """

        # Call the parent constructor.
        super().__init__()

        ## @var lin_net
        # Linear layers that process the continuous and discrete latent variables.
        self.lin_net = nn.Sequential(
            nn.Linear(n_continuous_vars + n_discrete_vars * n_discrete_vals, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11 * 11 * 64),
            nn.ReLU()
        )

        ## @var stack_size
        # Number of stacked frames in each observation.
        self.stack_size = relab.config("stack_size") if stack_size is None else stack_size

        ## @var up_conv_net
        # Transposed convolution layers that upscale the feature maps to the final image.
        self.up_conv_net = nn.Sequential(
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, (3, 3), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, self.stack_size, (3, 3), stride=(1, 1), padding=(3, 3), output_padding=(0, 0))
        )

    def forward(self, x : Tensor) -> Tensor:
        """!
        Perform the forward pass through the network.
        @param x: the observation
        @return the reconstructed image
        """
        x = self.lin_net(x)
        x = torch.reshape(x, (x.shape[0], 64, 11, 11))
        return self.up_conv_net(x)
