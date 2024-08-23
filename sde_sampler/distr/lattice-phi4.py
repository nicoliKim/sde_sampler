from __future__ import annotations

import torch

from .base import Distribution


class Phi4Distr(Distribution):
    r"""Action (unnormalized logprob) for the real-valued :math:`\phi^4` scalar field Theory in 1+1 dimensions.
    For reference see eq. :math:`S = \dots`, in https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.126.032001.
    It has two free parameters the bare coupling :math:`\lambda` and the hopping parameter :math:`\kappa`.
    The action reads

    .. math::

        S(\phi) = \sum_{x \in \Lambda} - 2 \kappa \sum_{\hat{\mu}=1}^2 \phi(x) \phi(x + \hat{\mu}) +
        (1 - 2 \lambda) \phi(x)^2 + \lambda \, \phi(x)^4 \,.

    The first sum runs over all sites :math:`x` of a lattice with volume :math:`\Lambda` and the second sum runs
    over nearest neighbours in two dimensions indicated by :math:`\hat{\mu}`.

    """
    def __init__(
        self,
        dim: int = 4,
        kappa: float = 0.3,
        lambd: float = 0.022,
        n_rows: int = 2,
        n_cols: int = 2,
        **kwargs,
    ):
        """Constructs all the necessary attributes for the `Phi4Action` action.

        Parameters
        ----------
        dim : int
            Dimensionality of the distribution.
        kappa : float
            The hopping parameter.
        lambd : float
            The bare coupling.

        """
        # if not dim == 2:
        #   raise ValueError(r"`dim` needs to be `2` for $\phi^4$-theory.")
        super().__init__(dim=4, **kwargs)
        self.kappa = kappa
        self.lambd = lambd
        self.n_rows = n_rows
        self.n_cols = n_cols

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Takes a batch of lattice configurations and evaluates the `Phi4Action` action in batches.

        Parameters
        ----------
        x : :py:obj:`torch.Tensor`
            Batch of the lattice configurations.

        Returns
        -------
        :py:obj:`torch.Tensor`
        Action evaluated at the given lattice configurations.

        Raises
        ------
        ValueError
            When the area of the rectangular lattice is not equal to the total number of points (d).

        """
        
        n_r, n_c = self.n_rows, self.n_cols
        kinetic, mass, inter = 0., 0., 0.
        action = torch.zeros(x.shape[0],1)
        
        if x.shape[1] != n_r * n_c:
            raise ValueError(f"Invalid lattice shape.\n "
                            f"Choose n_rows and n_cols in such a way that n_rows*n_cols={x.shape[1]}.")
        
        for i_config in range(x.shape[0]):
            lattice_x = x[i_config].view(n_r, n_c)
            for row in range(n_r):
                for col in range(n_c):
                    phi = lattice_x[row, col]                               #value of the field at the point (row,col)
                    mass += (1 - 2 * self.lambd) * phi ** 2                  #mass term at this point
                    inter = self.lambd * phi ** 4                           #interaction term at this point
                    
                    # Right neighbour
                    if col + 1 < n_c:
                        phi_right = lattice_x[row, (col + 1) % n_c]
                        kinetic += (-2 * self.kappa) * phi * phi_right      #kinetic contribution to the right
                    
                    # Down neighbor
                    if row + 1 < n_r:
                        phi_down = lattice_x[(row +1) % n_r, col]
                        kinetic += (-2 * self.kappa) * phi * phi_down       #kinetic contribution downwards
            
            action[i_config] = kinetic + mass + inter
        return action

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Computes the derivative of the phi-4 action, i.e., the score term for the Boltzmann distribution.

        Parameters
        ----------
        x : :py:obj:`torch.Tensor`
            Batch of the lattice configurations.

        Returns
        -------
        :py:obj:`torch.Tensor`
        Score function (grad of unnormalized log prob) evaluated at the given lattice configurations.

        """
        
        kinetic, mass, inter = 0., 0., 0.
        score = torch.zeros(x.shape)
        #Since the interaction between neighbors is quadratic, the derivative of the action is actually a free theory
        #This means that there's no need to specify any geometry
        
        for i_config in range(x.shape[0]):
            phi = x[i_config,:]
            mass = 2 * (1 - 2 * self.lambd) * phi        #derivative of the mass term at this point
            inter = 4 * self.lambd * phi ** 3            #derivative of the interaction term at this point
            kinetic = (-2 * self.kappa) * phi            #derivative of the kinetic term at this point
            score[i_config] = kinetic + mass + inter
        return score