from typing import Any, Optional

import torch
from tqdm import tqdm

from hydrodl2.models.hbv.hbv_2 import Hbv_2
from hydrodl2.models.hbv.hbv_2_1_hourly import Hbv_2_1_hourly
from hydrodl2.models.hbv.hbv_2_hourly import Hbv_2_hourly


class Hbv_2_mts(torch.nn.Module):
    """Multi-Timescale (MTS) HBV 2.0/2.1.

    Multi-component, multi-scale, differentiable PyTorch HBV model with rainfall
    runoff simulation on unit basins. Couples a low-frequency (daily) model with
    a high-frequency (hourly) model: daily model is spun-up to set the hourly
    model's initial states.

    Authors
    -------
    -   Wencong Yang, Leo Lonzarich
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005
        (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).

    Publication
    -----------
    -   Yang, W., Lonzarich, L., Song, Y., Ji, H., Pan, M., Lawson, K., & Shen,
        C. (2026). Hourly U.S.-Wide Flood Simulation beyond the Limits of
        Traditional and Data-Driven Models. Arxiv.
        https://arxiv.org/pdf/2609.06794 **[In Review]**

    Parameters
    ----------
    config
        The 'phy' section of the model config, with nested 'lof_model' and
        'hif_model' sub-configs for the low- and high-frequency models.
    device
        Device to run the model on.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'HBV 2.0/2.1 MTS'
        self.lof_config = config.get('lof_model', None)
        self.hif_config = config.get('hif_model', None)
        self._state_cache = [None, None]
        self.states = (None, None)
        self.dtype = torch.float32
        self.device = device

        # Unit basins and timesteps per sub-batch.
        self.train_spatial_chunk_size = 32768
        self.simulate_spatial_chunk_size = 10000
        self.simulate_temporal_chunk_size = 168
        self.simulate_mode = False

        # Warmup steps for routing during training.
        self.train_warmup = 168

        # State sideloading controls.
        self.load_from_cache = False
        self.use_from_cache = False

        # Low-frequency (daily) spin-up control.
        #   lof_rollout=False (default): the daily model is re-initialized
        #       before every forward, i.e. it is re-spun over the whole window
        #       it is given.
        #   lof_rollout=True: the daily model continues from its cached
        #       states, so a caller can spin it up once over a long window and
        #       then advance it over short ones. Requires
        #       `lof_model.cache_states = True`.
        self.lof_rollout = False

        # Reuse descaled static parameters; only for inference.
        self.cache_static_params = False
        self._static_param_cache: Optional[tuple] = None

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.hif_config is not None:
            # Overwrite defaults with config values.
            self.train_spatial_chunk_size = self.hif_config.get(
                'train_spatial_chunk_size',
                self.train_spatial_chunk_size,
            )
            self.simulate_spatial_chunk_size = self.hif_config.get(
                'simulate_spatial_chunk_size',
                self.simulate_spatial_chunk_size,
            )
            self.simulate_temporal_chunk_size = self.hif_config.get(
                'simulate_temporal_chunk_size',
                self.simulate_temporal_chunk_size,
            )
            self.train_warmup = self.hif_config.get(
                'train_warmup',
                self.train_warmup,
            )
        self.spatial_chunk_size = self.train_spatial_chunk_size

        self.lof_model = Hbv_2(self.lof_config, device=self.device)
        self.lof_model.initialize = True

        # TODO: could use better versioning here.
        hif_name = 'Hbv_2_hourly'
        if self.hif_config is not None:
            hif_name = self.hif_config.get('name', [hif_name])[0]
        if hif_name == 'Hbv_2_1_hourly':
            self.hif_model = Hbv_2_1_hourly(self.hif_config, device=self.device)
        elif hif_name == 'Hbv_2_hourly':
            self.hif_model = Hbv_2_hourly(self.hif_config, device=self.device)
        else:
            raise ValueError(f"High-frequency model '{hif_name}' not supported.")

        # Identity state transfer
        self.state_transfer_model = torch.nn.ModuleDict(
            {name: torch.nn.Identity() for name in self.hif_model.state_names}
        )

    def get_states(
        self,
    ) -> tuple[Optional[tuple[torch.Tensor, ...]], Optional[tuple[torch.Tensor, ...]]]:
        """States cached by the last forward pass of each submodel.

        Returns
        -------
        tuple[tuple, tuple]
            Tuple of (low-frequency, high-frequency) state tuples.
        """
        lof_states = self.lof_model.get_states()
        hif_states = self.hif_model.get_states()
        return (lof_states, hif_states)

    def load_states(
        self,
        state_tuple: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ) -> None:
        """Load internal model states, sideloading low-frequency states.

        Parameters
        ----------
        state_tuple
            Tuple of (low-frequency, high-frequency) state tuples.
        """
        if not isinstance(state_tuple, tuple) or len(state_tuple) != 2:
            raise ValueError("`states` must be a tuple of two tuples of tensors.")
        self._state_cache = [
            tuple(s[-1].detach().to(self.device, dtype=self.dtype) for s in states)
            for states in state_tuple
        ]

        if self.load_from_cache:
            # Only sideload low-frequency states.
            self.lof_model.load_states(state_tuple[0])

    def warmup_lof(
        self,
        x_dict: dict[str, torch.Tensor],
        lof_parameters: list[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Advance the low-frequency model and return its transferred states.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        lof_parameters
            Unprocessed, learned low-frequency parameters from a neural network.

        Returns
        -------
        tuple[torch.Tensor, ...]
            The transferred high-frequency initial states.
        """
        lof_x_dict = {
            'x_phy': x_dict['x_phy_lof'],
            'ac_all': x_dict['ac_all'],
            'elev_all': x_dict['elev_all'],
            'muwts': x_dict.get('muwts', None),
        }

        if not (self.lof_rollout and self.lof_model.cache_states):
            self.lof_model.states = None

        self.lof_model(lof_x_dict, lof_parameters)

        self._state_cache[0] = self.lof_model.states
        states = tuple(
            state.detach() for state in self.state_transfer(self.lof_model.states)
        )
        self._state_cache[1] = states
        self.states = (self._state_cache[0], states)

        return states

    def _forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Base forward.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Tuple of unprocessed, learned (low-frequency, high-frequency)
            parameters from a neural network.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of model outputs.
        """
        # 1. Transfer states
        lof_parameters, hif_parameters = parameters

        if self.use_from_cache and (self._state_cache[1] is not None):
            states = self.states[1]
        else:
            lof_x_dict = {
                'x_phy': x_dict['x_phy_lof'],
                'ac_all': x_dict['ac_all'],
                'elev_all': x_dict['elev_all'],
                'muwts': x_dict.get('muwts', None),
            }

            if not (self.lof_rollout and self.lof_model.cache_states):
                # Cold-start the daily model over the window it was given.
                self.lof_model.states = None

            self.lof_model(
                lof_x_dict,
                lof_parameters,
            )

            # Low-frequency states at last timestep
            self._state_cache[0] = self.lof_model.states
            states = self.state_transfer(self.lof_model.states)

        # 2. Transfer parameters
        phy_dy_params_dict, phy_static_params_dict, distr_params_dict = (
            self.param_transfer(
                lof_parameters,
                hif_parameters,
            )
        )

        # Run the model
        x = x_dict['x_phy_hif']

        ac = x_dict['ac_all'].unsqueeze(-1).expand(-1, self.hif_model.nmul)
        elevation = x_dict['elev_all'].unsqueeze(-1).expand(-1, self.hif_model.nmul)
        outlet_topo = x_dict['outlet_topo']
        areas = x_dict['areas']

        predictions, hif_states = self.hif_model._PBM(
            forcing=x,
            ac=ac,
            elevation=elevation,
            states=tuple(states),
            phy_dy_params_dict=phy_dy_params_dict,
            phy_static_params_dict=phy_static_params_dict,
            outlet_topo=outlet_topo,
            areas=areas,
            distr_params_dict=distr_params_dict,
        )

        # State caching
        self._state_cache[1] = tuple(s.detach() for s in hif_states)
        if self.load_from_cache:
            new_states = []

            # Low-frequency states remain the same.
            new_states.append(self._state_cache[0])

            # High-frequency states updated.
            new_states.append(tuple(s[-1] for s in hif_states))
            self.states = tuple(new_states)

        return predictions

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Forward pass supporting spatial and temporal chunking.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Tuple of unprocessed, learned (low-frequency, high-frequency)
            parameters from a neural network.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of model outputs.
        """
        device = self.device
        n_units = x_dict['areas'].shape[0]
        spatial_chunk_size = self.spatial_chunk_size
        temporal_chunk_size = self.simulate_temporal_chunk_size
        train_warmup = self.train_warmup

        if (not self.simulate_mode) and (n_units <= spatial_chunk_size):
            self.hif_model.use_distr_routing = False
            return self._forward(x_dict, parameters)

        # Chunked runoff generation for simulation mode or large training batches.
        self.hif_model.use_distr_routing = False
        preds_list = []
        prog_bar = tqdm(
            range(0, n_units, spatial_chunk_size),
            desc="Spatial runoff chunks",
        )

        for i in prog_bar:
            end_idx = min(i + spatial_chunk_size, n_units)
            reach_idx = (x_dict['outlet_topo'] == 1).nonzero(as_tuple=False)
            idxs_in_chunk = (reach_idx[:, 1] >= i) & (reach_idx[:, 1] < end_idx)

            chunk_x_dict = {
                'x_phy_lof': x_dict['x_phy_lof'][:, i:end_idx].to(device),
                'x_phy_hif': x_dict['x_phy_hif'][:, i:end_idx].to(device),
                'ac_all': x_dict['ac_all'][i:end_idx].to(device),
                'elev_all': x_dict['elev_all'][i:end_idx].to(device),
                'areas': x_dict['areas'][i:end_idx].to(device),
                'outlet_topo': x_dict['outlet_topo'][:, i:end_idx].to(device),
            }
            chunk_parameters = (
                [
                    parameters[0][0][:, i:end_idx].to(
                        device
                    ),  # low-freq dynamic phy params
                    parameters[0][1][i:end_idx].to(
                        device
                    ),  # low-freq static phy params
                ],
                [
                    parameters[1][0][:, i:end_idx].to(
                        device
                    ),  # high-freq dynamic phy params
                    parameters[1][1][i:end_idx].to(
                        device
                    ),  # high-freq static phy params
                    parameters[1][2][idxs_in_chunk].to(
                        device
                    ),  # high-freq distributed params
                ],
            )
            chunk_predictions = self._forward(chunk_x_dict, chunk_parameters)
            preds_list.append(chunk_predictions)

        predictions = self.concat_spatial_chunks(preds_list)
        runoff = predictions['Qs']
        hif_length = runoff.shape[0]

        # Chunked routing
        _, _, _, distr_params = self.hif_model._unpack_parameters(parameters[1])
        distr_params_dict = self.hif_model._descale_distr_parameters(distr_params)
        distr_params_dict = {
            key: value.to(device) for key, value in distr_params_dict.items()
        }
        outlet_topo = x_dict['outlet_topo'].to(device)
        areas = x_dict['areas'].to(device)

        preds_list = []
        prog_bar = tqdm(
            range(train_warmup, hif_length, temporal_chunk_size),
            desc="Temporal routing chunks",
        )

        for t in prog_bar:
            end_t = min(t + temporal_chunk_size, hif_length)
            chunk_runoff = runoff[t - train_warmup : end_t]
            chunk_predictions = self.hif_model.distr_routing(
                Qs=chunk_runoff,
                distr_params_dict=distr_params_dict,
                outlet_topo=outlet_topo,
                areas=areas,
            )

            # Remove routing warmup for all but first chunk
            if t > train_warmup:
                chunk_predictions = {
                    key: value[train_warmup:]
                    for key, value in chunk_predictions.items()
                }
            preds_list.append(chunk_predictions)

        routing_predictions = self.concat_temporal_chunks(preds_list)
        predictions['streamflow'] = routing_predictions['Qs_rout']

        return predictions

    def set_mode(self, is_simulate: bool) -> None:
        """Set simulate mode.

        Parameters
        ----------
        is_simulate
            If True, use the simulation spatial chunk size; otherwise use the
            training spatial chunk size.
        """
        if is_simulate:
            self.spatial_chunk_size = self.simulate_spatial_chunk_size
            self.simulate_mode = True
        else:
            self.spatial_chunk_size = self.train_spatial_chunk_size
            self.simulate_mode = False

    def param_transfer(
        self,
        lof_parameters: list[torch.Tensor],
        hif_parameters: list[torch.Tensor],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        """Map low-frequency parameters to high-frequency parameters.

        Static parameters shared with the low-frequency model are carried over
        from it; dynamic, distributed, and high-frequency-only static parameters
        are taken from the high-frequency model.

        Parameters
        ----------
        lof_parameters
            Unprocessed, learned low-frequency parameters from a neural network.
        hif_parameters
            Unprocessed, learned high-frequency parameters from a neural
            network.

        Returns
        -------
        tuple[dict, dict, dict]
            Tuple of descaled dynamic, static, and distributed parameter
            dictionaries for the high-frequency model.
        """
        phy_dy_params, phy_static_params, routing_params, distr_params = (
            self.hif_model._unpack_parameters(hif_parameters)
        )
        # New dynamic parameters
        phy_dy_params_dict = self.hif_model._descale_phy_dy_parameters(
            phy_dy_params, dy_list=self.hif_model.dynamic_params
        )

        cached = self._static_param_cache
        if (
            self.cache_static_params
            and cached is not None
            and cached[0] is lof_parameters[1]
            and cached[1] is hif_parameters[1]
            and cached[2] is hif_parameters[2]
        ):
            phy_static_params_dict, distr_params_dict, routing_param_dict = cached[3:]
            if self.hif_model.routing:
                self.hif_model.routing_param_dict = routing_param_dict
            return phy_dy_params_dict, phy_static_params_dict, distr_params_dict

        _, warmup_phy_static_params, _ = self.lof_model._unpack_parameters(
            lof_parameters
        )

        # Keep warmup static params, add high-freq specific static parameters
        static_param_names = [
            param
            for param in self.hif_model.phy_param_names
            if param not in self.hif_model.dynamic_params
        ]
        warmup_static_param_names = [
            param
            for param in self.lof_model.phy_param_names
            if param not in self.lof_model.dynamic_params
        ]
        var_indexes = [
            i
            for i, param in enumerate(static_param_names)
            if param not in warmup_static_param_names
        ]

        n_warmup = len(warmup_static_param_names)
        if static_param_names[:n_warmup] != warmup_static_param_names:
            raise ValueError(
                "Static parameter order mismatch between low- and "
                "high-frequency models: expected the first "
                f"{n_warmup} of {static_param_names} to be "
                f"{warmup_static_param_names}. Shared static parameters must "
                "appear in the same order in both models.",
            )

        phy_static_params_dict = self.hif_model._descale_phy_stat_parameters(
            torch.concat(
                [warmup_phy_static_params, phy_static_params[:, var_indexes]], dim=1
            ),
            stat_list=static_param_names,
        )
        # New distributed parameters
        distr_params_dict = self.hif_model._descale_distr_parameters(distr_params)

        # New routing params
        routing_param_dict = None
        if self.hif_model.routing:
            routing_param_dict = self.hif_model._descale_route_parameters(
                routing_params
            )
            self.hif_model.routing_param_dict = routing_param_dict

        if self.cache_static_params:
            self._static_param_cache = (
                lof_parameters[1],
                hif_parameters[1],
                hif_parameters[2],
                phy_static_params_dict,
                distr_params_dict,
                routing_param_dict,
            )

        return phy_dy_params_dict, phy_static_params_dict, distr_params_dict

    def state_transfer(self, states: tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Map low-frequency states to high-frequency states.

        Parameters
        ----------
        states
            Low-frequency states, ordered as `lof_model.state_names`.

        Returns
        -------
        list[torch.Tensor]
            The transferred high-frequency states.
        """
        states_dict = dict(zip(self.lof_model.state_names, states))
        return [
            self.state_transfer_model[key](states_dict[key])
            for key in self.lof_model.state_names
        ]

    @staticmethod
    def concat_spatial_chunks(
        pred_list: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Concatenate spatial chunk predictions.

        Parameters
        ----------
        pred_list
            List of per-chunk model output dictionaries.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of concatenated model outputs.
        """
        output = {}
        for key in pred_list[0].keys():
            if pred_list[0][key].ndim == 3:
                output[key] = torch.cat(
                    [preds[key] for preds in pred_list], dim=1
                )  # (window_size, n_units, nmul)
            else:
                output[key] = torch.cat(
                    [preds[key] for preds in pred_list], dim=0
                )  # (n_units, nmul) or (n_units,)
        return output

    @staticmethod
    def concat_temporal_chunks(
        pred_list: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Concatenate temporal chunk predictions.

        Parameters
        ----------
        pred_list
            List of per-chunk model output dictionaries.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary of concatenated model outputs.
        """
        output = {}
        for key in pred_list[0].keys():
            if pred_list[0][key].ndim == 3:
                output[key] = torch.cat(
                    [preds[key] for preds in pred_list], dim=0
                )  # (window_size, n, nmul)
            else:
                output[key] = pred_list[0][key]  # (n_units, nmul) or (n_units,)
        return output
