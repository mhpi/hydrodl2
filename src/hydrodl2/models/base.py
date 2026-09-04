from typing import Any, Optional

import torch

from hydrodl2.core.calc import change_param_range, trim_warmup


class BasePhysicsModel(torch.nn.Module):
    """Common code for hydrodl2-based models.

    Subclasses keep their own `__init__` and should set their own
    1. state_names
    2. nmul
    3. device
    4. warmup
    5. warmup_states
    6. parameter_bounds
    7. routing_parameter_bounds

    Forward: returns a timeseries dict (nsteps - warmup); warmup is stripped
    inside the model -- either simulate it separately (warmup_states=True) or
    run the full time window and drop the lead.
    """

    #: Nonzero initial states for safe powers/divisions.
    initial_state_value: float = 0.001

    def __init__(self) -> None:
        super().__init__()
        self.states: Optional[tuple[torch.Tensor, ...]] = None
        self._state_cache: Optional[tuple[torch.Tensor, ...]] = None

    @staticmethod
    def trim_warmup(
        outputs: dict[str, torch.Tensor],
        pred_cutoff: int,
        nsteps: int,
    ) -> dict[str, torch.Tensor]:
        """Drop `pred_cutoff` leading steps from timeseries outputs."""
        return trim_warmup(outputs, pred_cutoff, nsteps)

    def _init_states(self, ngrid: int) -> tuple[torch.Tensor, ...]:
        """One [ngrid, nmul] tensor per state with initial_state_value."""

        def make_state():
            return torch.full(
                (ngrid, self.nmul),
                self.initial_state_value,
                dtype=torch.float32,
                device=self.device,
            )

        return tuple(make_state() for _ in range(len(self.state_names)))

    def get_states(self) -> Optional[tuple[torch.Tensor, ...]]:
        """States cached by the last forward pass, or None if no yet run."""
        return self._state_cache

    def load_states(self, states: tuple[torch.Tensor, ...]) -> None:
        """Load states into the model."""
        for state in states:
            if not isinstance(state, torch.Tensor):
                raise ValueError("Each element in states must be a tensor.")
        nstates = len(self.state_names)
        if not (isinstance(states, tuple) and len(states) == nstates):
            raise ValueError(f"States must be a tuple of {nstates} tensors.")

        self.states = tuple(
            s.detach().to(self.device, dtype=torch.float32) for s in states
        )

    def _descale_route_parameters(
        self,
        routing_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map normalized routing parameters onto their physical ranges.
        
        Shape: [ngrid, n_route, nmul]

        Parameters
        ----------
        routing_params
            Normalized routing parameters.
        
        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(self.routing_parameter_bounds.keys()):
            parameter_dict[name] = change_param_range(
                param=routing_params[:, i],
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def _descale_phy_dy_parameters(
        self,
        phy_dy_params: torch.Tensor,
        dy_list: list[str],
    ) -> dict[str, torch.Tensor]:
        """Descale the time-dynamic physical parameters.
        
        Shape: [nsteps, ngrid, n_dynamic, nmul]

        Parameters
        ----------
        phy_dy_params
            Normalized dynamic physical parameters.
        dy_list
            List of dynamic parameters.

        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        raise NotImplementedError

    def _descale_phy_stat_parameters(
        self,
        phy_stat_params: torch.Tensor,
        stat_list: list[str],
    ) -> dict[str, torch.Tensor]:
        """Descale the time-invariant physical parameters.
        
        Shape: [ngrid, n_static, nmul]

        Parameters
        ----------
        phy_stat_params
            Normalized static physical parameters.

        Returns
        -------
        dict
            Dictionary of descaled static physical parameters.
        """
        raise NotImplementedError

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: Any,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.
        
        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of model outputs.
        """
        raise NotImplementedError
