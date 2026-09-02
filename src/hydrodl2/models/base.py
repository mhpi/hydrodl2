"""Shared base class for HydroDL2 process-based (physics) models."""

from typing import Any, Optional

import torch

from hydrodl2.core.calc import change_param_range, trim_warmup


class BasePhysicsModel(torch.nn.Module):
    """Common scaffolding for HydroDL2 process-based models.

    Subclasses keep their own ``__init__`` and only need to call
    ``super().__init__()``. Everything here reads attributes the subclass
    sets, so there is no fixed constructor signature to conform to.

    The forward contract
    --------------------
    ``forward(x_dict, parameters)`` returns a dict of model outputs whose
    time-major entries cover exactly ``nsteps - self.warmup`` timesteps.
    Warm-up is stripped *inside the model*, never by the caller. Two
    strategies satisfy this and both are supported:

    - ``warmup_states=True``: simulate the warm-up window separately under
      ``torch.no_grad()`` to spin up storages, then run the remaining window.
      The warm-up steps never enter the outputs, so nothing needs trimming.
    - ``warmup_states=False``: simulate the whole window, then drop the
      leading ``self.warmup`` steps with :meth:`trim_warmup` before returning.

    Whichever is used, a caller can rely on the returned length. Models that
    do not implement a separate spin-up pass should always take the second
    path and say so in their class docstring.

    Attributes a subclass is expected to set
    ---------------------------------------
    state_names
        Names of the internal storages, in the order ``_PBM`` returns them.
    nmul
        Number of parallel model instances per basin.
    device
        Device the model runs on.
    warmup, warmup_states
        Warm-up length and strategy, described above.
    parameter_bounds, routing_parameter_bounds
        Physical and routing parameter ranges, as ``{name: [lower, upper]}``.
    """

    #: Value every internal storage is initialized to. Zero is avoided so
    #: that divisions and powers taken on a fresh storage stay finite.
    initial_state_value: float = 0.001

    def __init__(self) -> None:
        super().__init__()
        self.states: Optional[tuple[torch.Tensor, ...]] = None
        self._state_cache: Optional[tuple[torch.Tensor, ...]] = None

    # ------------------------------------------------------------------ #
    #  Warm-up                                                           #
    # ------------------------------------------------------------------ #
    def _resolve_warmup(self) -> tuple[int, int]:
        """Resolve the warm-up strategy into concrete step counts.

        Returns
        -------
        tuple[int, int]
            ``(spinup_steps, pred_cutoff)``. ``spinup_steps`` is how many
            leading steps to simulate separately before the scored window;
            ``pred_cutoff`` is how many leading steps to drop from the
            outputs afterwards. Exactly one of them is non-zero.
        """
        if getattr(self, 'warmup_states', True):
            return self.warmup, 0
        return 0, self.warmup

    @staticmethod
    def trim_warmup(
        outputs: dict[str, torch.Tensor],
        pred_cutoff: int,
        nsteps: int,
    ) -> dict[str, torch.Tensor]:
        """Drop ``pred_cutoff`` leading steps from every time-major output.

        Outputs that have already collapsed the time axis (a baseflow index
        summed over time, say) are passed through untouched, so subclasses do
        not have to maintain a list of exceptions.
        """
        return trim_warmup(outputs, pred_cutoff, nsteps)

    # ------------------------------------------------------------------ #
    #  Internal states                                                   #
    # ------------------------------------------------------------------ #
    def _init_states(self, ngrid: int) -> tuple[torch.Tensor, ...]:
        """Initialize every internal storage to :attr:`initial_state_value`.

        Parameters
        ----------
        ngrid
            Number of basins/catchments in the batch.

        Returns
        -------
        tuple[torch.Tensor, ...]
            One ``[ngrid, nmul]`` tensor per entry in ``state_names``.
        """

        def make_state():
            return torch.full(
                (ngrid, self.nmul),
                self.initial_state_value,
                dtype=torch.float32,
                device=self.device,
            )

        return tuple(make_state() for _ in range(len(self.state_names)))

    def get_states(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Return the internal states cached by the last forward pass.

        Returns
        -------
        tuple[torch.Tensor, ...] or None
            One tensor per entry in ``state_names``, or ``None`` if the model
            has not been run yet.
        """
        return self._state_cache

    def load_states(self, states: tuple[torch.Tensor, ...]) -> None:
        """Load internal states, moved to the model's device and dtype.

        Parameters
        ----------
        states
            One tensor per entry in ``state_names``.
        """
        for state in states:
            if not isinstance(state, torch.Tensor):
                raise ValueError("Each element in `states` must be a tensor.")
        nstates = len(self.state_names)
        if not (isinstance(states, tuple) and len(states) == nstates):
            raise ValueError(f"`states` must be a tuple of {nstates} tensors.")

        self.states = tuple(
            s.detach().to(self.device, dtype=torch.float32) for s in states
        )

    # ------------------------------------------------------------------ #
    #  Parameter descaling                                               #
    # ------------------------------------------------------------------ #
    def _descale_route_parameters(
        self,
        routing_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Map normalized routing parameters onto their physical ranges.

        Parameters
        ----------
        routing_params
            Normalized routing parameters, ``[ngrid, n_routing]``.

        Returns
        -------
        dict[str, torch.Tensor]
            Descaled routing parameters keyed by name.
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
        """Descale the time-varying physical parameters.

        Split from the static parameters on purpose: only these carry a time
        axis, so the network emitting them can be much narrower. Keeping the
        static ones out of this tensor is what lets distributed models run on
        large catchment counts.

        Parameters
        ----------
        phy_dy_params
            Normalized dynamic parameters, ``[nsteps, ngrid, n_dynamic, nmul]``.
        dy_list
            Names of the dynamic parameters, in channel order.
        """
        raise NotImplementedError

    def _descale_phy_stat_parameters(
        self,
        phy_stat_params: torch.Tensor,
        stat_list: list[str],
    ) -> dict[str, torch.Tensor]:
        """Descale the time-invariant physical parameters.

        Parameters
        ----------
        phy_stat_params
            Normalized static parameters, ``[ngrid, n_static, nmul]`` — note
            the absence of a time axis.
        stat_list
            Names of the static parameters, in channel order.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Required of every model                                           #
    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the model. See the class docstring for the return contract."""
        raise NotImplementedError
