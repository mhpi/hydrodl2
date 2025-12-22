from typing import Any, Optional

import torch
from tqdm import tqdm

from hydrodl2.models.hbv.hbv_2 import Hbv_2
from hydrodl2.models.hbv.hbv_2_hourly import Hbv_2_hourly


class Hbv_2_mts(torch.nn.Module):
    """HBV 2.0, multi timescale, distributed UH.

    Multi-component, multi-scale, differentiable PyTorch HBV model with rainfall
    runoff simulation on unit basins.

    Authors
    -------
    -   Wencong Yang
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005
        (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        Device to run the model on.
    """

    def __init__(
        self,
        low_freq_config: Optional[dict[str, Any]] = None,
        high_freq_config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = torch.float32
        self.low_freq_model = Hbv_2(low_freq_config, device=device)
        self.low_freq_model.initialize = True
        self.high_freq_model = Hbv_2_hourly(high_freq_config, device=device)
        self._state_cache = [None, None]
        self.states = (None, None)
        self.load_from_cache = False
        self.use_from_cache = False

        # # learnable transfer
        # self.state_transfer_model = torch.nn.ModuleDict(
        #     {
        #         name: torch.nn.Sequential(
        #             torch.nn.Linear(
        #                 self.low_freq_model.nmul, self.high_freq_model.nmul
        #             ),
        #             torch.nn.ReLU(),
        #         )
        #         for name in self.high_freq_model.state_names
        #     }
        # )
        # Identity state transfer
        self.state_transfer_model = torch.nn.ModuleDict(
            {name: torch.nn.Identity() for name in self.high_freq_model.state_names}
        )

        self.train_spatial_chunk_size = high_freq_config['train_spatial_chunk_size']
        self.simulate_spatial_chunk_size = high_freq_config[
            'simulate_spatial_chunk_size'
        ]
        self.simulate_temporal_chunk_size = high_freq_config[
            'simulate_temporal_chunk_size'
        ]
        self.spatial_chunk_size = self.train_spatial_chunk_size
        self.simulate_mode = False

        # warmup steps for routing during training.
        self.train_warmup = high_freq_config['train_warmup']

    def get_states(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Return internal states for high and low frequency models."""
        lof_states = self.low_freq_model.get_states()
        hif_states = self.high_freq_model.get_states()
        return (lof_states, hif_states)

    def load_states(
        self,
        state_tuple: tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]],
    ) -> None:
        """Load internal model states sideload low frequency states."""
        if not isinstance(state_tuple, tuple) or len(state_tuple) != 2:
            raise ValueError("`states` must be a tuple of two tuples of tensors.")
        self._state_cache = tuple(
            tuple(s[-1].detach().to(self.device, dtype=self.dtype) for s in states)
            for states in state_tuple
        )

        if self.load_from_cache:
            # Only sideload low-frequency states.
            self.low_freq_model.load_states(state_tuple[0])

    def _forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Base forward."""
        # 1. Transfer states
        low_freq_parameters, high_freq_parameters = parameters

        if self.use_from_cache and (self._state_cache[1] is not None):
            states = self.states[1]
        else:
            low_freq_x_dict = {
                'x_phy': x_dict['x_phy_low_freq'],
                'ac_all': x_dict['ac_all'],
                'elev_all': x_dict['elev_all'],
                'muwts': x_dict.get('muwts', None),
            }

            self.low_freq_model.states = None
            self.low_freq_model(
                low_freq_x_dict,
                low_freq_parameters,
            )

            # Low-frequency states at last timestep
            self._state_cache[0] = self.low_freq_model.states
            states = self.state_transfer(self.low_freq_model.states)

        # 2. Transfer parameters
        phy_dy_params_dict, phy_static_params_dict, distr_params_dict = (
            self.param_transfer(
                low_freq_parameters,
                high_freq_parameters,
            )
        )

        # Run the model
        x = x_dict['x_phy_high_freq']

        Ac = x_dict['ac_all'].unsqueeze(-1).expand(-1, self.high_freq_model.nmul)
        Elevation = (
            x_dict['elev_all'].unsqueeze(-1).expand(-1, self.high_freq_model.nmul)
        )
        outlet_topo = x_dict['outlet_topo']
        areas = x_dict['areas']

        predictions, hif_states = self.high_freq_model._PBM(
            forcing=x,
            Ac=Ac,
            Elevation=Elevation,
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

            # low-frequency states remain the same
            new_states.append(self._state_cache[0])

            # high-frequency states updated
            new_states.append(tuple(s[-1] for s in hif_states))
            self.states = tuple(new_states)

        # Temp: save initial states
        # torch.save(tuple(tuple(s.detach().cpu() for s in states) for states in self._state_cache), "/projects/mhpi/leoglonz/ciroh-ua/dhbv2_mts/ngen_resources/data/dhbv2_mts/models/hfv2.2_15yr/initial_states_2009.pt")

        return predictions

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Foward supports spatial and temporal chunking.

        x_dict and parameters can be in cpu for simulation mode to save GPU
        memory.
        """
        device = self.device
        n_units = x_dict['areas'].shape[0]
        spatial_chunk_size = self.spatial_chunk_size
        temporal_chunk_size = self.simulate_temporal_chunk_size
        train_warmup = self.train_warmup

        if (not self.simulate_mode) and (n_units <= spatial_chunk_size):
            self.high_freq_model.use_distr_routing = False
            return self._forward(x_dict, parameters)

        # Chunked runoff generation for simulation mode or large training batches
        self.high_freq_model.use_distr_routing = False
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
                'x_phy_low_freq': x_dict['x_phy_low_freq'][:, i:end_idx].to(device),
                'x_phy_high_freq': x_dict['x_phy_high_freq'][:, i:end_idx].to(device),
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
        high_freq_length = runoff.shape[0]

        # Chunked routing
        _, _, _, distr_params = self.high_freq_model.unpack_parameters(parameters[1])
        distr_params_dict = self.high_freq_model._descale_distr_parameters(distr_params)
        distr_params_dict = {
            key: value.to(device) for key, value in distr_params_dict.items()
        }
        outlet_topo = x_dict['outlet_topo'].to(device)
        areas = x_dict['areas'].to(device)

        preds_list = []
        prog_bar = tqdm(
            range(train_warmup, high_freq_length, temporal_chunk_size),
            desc="Temporal routing chunks",
        )

        for t in prog_bar:
            end_t = min(t + temporal_chunk_size, high_freq_length)
            chunk_runoff = runoff[t - train_warmup : end_t]
            chunk_predictions = self.high_freq_model.distr_routing(
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

    def set_mode(self, is_simulate: bool):
        """Set simulate mode."""
        if is_simulate:
            self.spatial_chunk_size = self.simulate_spatial_chunk_size
            self.simulate_mode = True
        else:
            self.spatial_chunk_size = self.train_spatial_chunk_size
            self.simulate_mode = False

    def param_transfer(
        self,
        low_freq_parameters: list[torch.Tensor],
        high_freq_parameters: list[torch.Tensor],
    ):
        """Map low-frequency parameters to high-frequency parameters."""
        warmup_phy_dy_params, warmup_phy_static_params, warmup_routing_params = (
            self.low_freq_model._unpack_parameters(low_freq_parameters)
        )

        phy_dy_params, phy_static_params, routing_params, distr_params = (
            self.high_freq_model._unpack_parameters(high_freq_parameters)
        )
        # New dynamic params
        phy_dy_params_dict = self.high_freq_model._descale_phy_dy_parameters(
            phy_dy_params, dy_list=self.high_freq_model.dynamic_params
        )

        # Keep warmup static params, add high-freq specific static params
        static_param_names = [
            param
            for param in self.high_freq_model.phy_param_names
            if param not in self.high_freq_model.dynamic_params
        ]
        warmup_static_param_names = [
            param
            for param in self.low_freq_model.phy_param_names
            if param not in self.low_freq_model.dynamic_params
        ]
        var_indexes = [
            i
            for i, param in enumerate(static_param_names)
            if param not in warmup_static_param_names
        ]
        phy_static_params_dict = self.high_freq_model._descale_phy_stat_parameters(
            torch.concat(
                [warmup_phy_static_params, phy_static_params[:, var_indexes]], dim=1
            ),
            stat_list=static_param_names,
        )
        # New distributed params
        distr_params_dict = self.high_freq_model._descale_distr_parameters(distr_params)

        # New routing params
        if self.high_freq_model.routing:
            self.high_freq_model.routing_param_dict = (
                self.high_freq_model._descale_rout_parameters(routing_params)
            )

        return phy_dy_params_dict, phy_static_params_dict, distr_params_dict

    def state_transfer(self, states: list[torch.Tensor]):
        """Map low-frequency states to high-frequency states."""
        states_dict = dict(zip(self.high_freq_model.state_names, states))
        return [
            self.state_transfer_model[key](states_dict[key])
            for key in states_dict.keys()
        ]

    @staticmethod
    def concat_spatial_chunks(pred_list: list[dict[str, torch.Tensor]]):
        """Concatenate spatial chunk pedictions."""
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
    def concat_temporal_chunks(pred_list: list[dict[str, torch.Tensor]]):
        """Concatenate temporal chunk predictions."""
        output = {}
        for key in pred_list[0].keys():
            if pred_list[0][key].ndim == 3:
                output[key] = torch.cat(
                    [preds[key] for preds in pred_list], dim=0
                )  # (window_size, n, nmul)
            else:
                output[key] = pred_list[0][key]  # (n_units, nmul) or (n_units,)
        return output
