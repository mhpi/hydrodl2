from typing import Any, Optional, Union

import torch

from hydrodl2.core.calc import change_param_range, uh_conv, uh_gamma


class Hbv_1_1p(torch.nn.Module):
    """HBV 1.1p.

    Multi-component, differentiable Pytorch HBV model with a capillary rise
    modification and option to run without internal state warmup.

    Authors
    -------
    -   Yalan Song, Farshid Rahmani, Leo Lonzarich
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005
        (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).

    Publication
    -----------
    -   Yalan Song, Kamlesh Sawadekar, Jonathan M Frame, et al. Physics-informed,
        Differentiable Hydrologic  Models for Capturing Unseen Extreme Events.
        ESS Open Archive. March 14, 2025.
        https://doi.org/10.22541/essoar.172304428.82707157/v2 **[Accepted]**

    Parameters
    ----------
    config
        Configuration dictionary.
    device
        Device to run the model on.
    """

    def __init__(
        self,
        config: Optional[dict[str, Any]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.name = 'HBV 1.1p'
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.pred_cutoff = 0
        self.warm_up_states = True
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = True
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.cache_states = False
        self.device = device

        self.states, self._states_cache = None, None

        self.state_names = [
            'SNOWPACK',  # Snowpack storage
            'MELTWATER',  # Meltwater storage
            'SM',  # Soil moisture storage
            'SUZ',  # Upper groundwater storage
            'SLZ',  # Lower groundwater storage
        ]
        self.flux_names = [
            'streamflow',  # Routed Streamflow
            'srflow',  # Routed surface runoff
            'ssflow',  # Routed subsurface flow
            'gwflow',  # Routed groundwater flow
            'AET_hydro',  # Actual ET
            'PET_hydro',  # Potential ET
            'SWE',  # Snow water equivalent
            'streamflow_no_rout',  # Streamflow
            'srflow_no_rout',  # Surface runoff
            'ssflow_no_rout',  # Subsurface flow
            'gwflow_no_rout',  # Groundwater flow
            'recharge',  # Recharge
            'excs',  # Excess stored water
            'evapfactor',  # Evaporation factor
            'tosoil',  # Infiltration
            'percolation',  # Percolation
            'capillary',  # Capillary rise
            'BFI',  # Baseflow index
        ]

        self.parameter_bounds = {
            'parBETA': [1.0, 6.0],
            'parFC': [50, 1000],
            'parK0': [0.05, 0.9],
            'parK1': [0.01, 0.5],
            'parK2': [0.001, 0.2],
            'parLP': [0.2, 1],
            'parPERC': [0, 10],
            'parUZL': [0, 100],
            'parTT': [-2.5, 2.5],
            'parCFMAX': [0.5, 10],
            'parCFR': [0, 0.1],
            'parCWH': [0, 0.2],
            'parBETAET': [0.3, 5],
            'parC': [0, 1],
        }
        self.routing_parameter_bounds = {
            'route_a': [0, 2.9],
            'route_b': [0, 6.5],
        }

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get(
                self.__class__.__name__, self.dynamic_params
            )
            self.variables = config.get('variables', self.variables)
            self.routing = config.get('routing', self.routing)
            self.comprout = config.get('comprout', self.comprout)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
            self.cache_states = config.get('cache_states', False)
        self._set_parameters()

    def _init_states(self, ngrid: int) -> tuple[torch.Tensor]:
        """Initialize model states to zero."""

        def make_state():
            return torch.full(
                (ngrid, self.nmul), 0.001, dtype=torch.float32, device=self.device
            )

        return tuple(make_state() for _ in range(len(self.state_names)))

    def get_states(self) -> Optional[tuple[torch.Tensor, ...]]:
        """Return internal model states.

        Returns
        -------
        tuple[torch.Tensor, ...]
            A tuple containing the states (SNOWPACK, MELTWATER, SM, SUZ, SLZ).
        """
        return self._states_cache

    def load_states(
        self,
        states: tuple[torch.Tensor, ...],
    ) -> None:
        """Load internal model states and set to model device and type.

        Parameters
        ----------
        states
            A tuple containing the states (SNOWPACK, MELTWATER, SM, SUZ, SLZ).
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

    def _set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        if self.routing:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []

        self.learnable_param_count = len(self.phy_param_names) * self.nmul + len(
            self.routing_param_names
        )

    def _unpack_parameters(
        self,
        parameters: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.

        Parameters
        ----------
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Tuple of physical and routing parameters.
        """
        phy_param_count = len(self.parameter_bounds)

        # Physical parameters
        phy_params = torch.sigmoid(
            parameters[:, :, : phy_param_count * self.nmul]
        ).view(
            parameters.shape[0],
            parameters.shape[1],
            phy_param_count,
            self.nmul,
        )
        # Routing parameters
        routing_params = None
        if self.routing:
            routing_params = torch.sigmoid(
                parameters[-1, :, phy_param_count * self.nmul :],
            )
        return (phy_params, routing_params)

    def _descale_phy_parameters(
        self,
        phy_params: torch.Tensor,
        dy_list: list,
    ) -> dict[str, torch.Tensor]:
        """Descale physical parameters.

        Parameters
        ----------
        phy_params
            Normalized physical parameters.
        dy_list
            List of dynamic parameters.

        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        nsteps = phy_params.shape[0]
        ngrid = phy_params.shape[1]

        param_dict = {}
        pmat = torch.ones([1, ngrid, 1]) * self.dy_drop
        for i, name in enumerate(self.parameter_bounds.keys()):
            staPar = phy_params[-1, :, i, :].unsqueeze(0).repeat([nsteps, 1, 1])
            if name in dy_list:
                dynPar = phy_params[:, :, i, :]
                drmask = torch.bernoulli(pmat).detach_().to(self.device)
                comPar = dynPar * (1 - drmask) + staPar * drmask
                param_dict[name] = change_param_range(
                    param=comPar,
                    bounds=self.parameter_bounds[name],
                )
            else:
                param_dict[name] = change_param_range(
                    param=staPar,
                    bounds=self.parameter_bounds[name],
                )
        return param_dict

    def _descale_route_parameters(
        self,
        routing_params: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Descale routing parameters.

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
            param = routing_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name],
            )
        return parameter_dict

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: torch.Tensor,
    ) -> Union[tuple, tuple[dict[str, torch.Tensor], tuple]]:
        """Forward pass.

        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        Union[tuple, tuple[dict, tuple]]
            Tuple or dictionary of model outputs.
        """
        # Unpack input data.
        x = x_dict['x_phy']
        self.muwts = x_dict.get('muwts', None)
        ngrid = x.shape[1]

        # Unpack parameters.
        phy_params, routing_params = self._unpack_parameters(parameters)
        if self.routing:
            self.routing_param_dict = self._descale_route_parameters(routing_params)

        # Initialization
        if self.warm_up_states:
            warm_up = self.warm_up
        else:
            # No state warm up: run the full model for warm_up days.
            self.pred_cutoff = self.warm_up
            warm_up = 0

        if (not self.states) or (not self.cache_states):
            current_states = self._init_states(ngrid)
        else:
            current_states = self.states

        # Warm-up model states - run the model only on warm_up days first.
        if warm_up > 0:
            with torch.no_grad():
                phy_param_warmup_dict = self._descale_phy_parameters(
                    phy_params[:warm_up, :, :],
                    dy_list=[],
                )
                # a. Save current model settings.
                init_flag, route_flag = self.initialize, self.routing

                # b. Set temporary model settings for warm-up.
                self.initialize, self.routing = True, False

                current_states = self._PBM(
                    x[:warm_up, :, :],
                    current_states,
                    phy_param_warmup_dict,
                )

                # c. Restore model settings.
                self.initialize, self.routing = init_flag, route_flag

        # Run the model for remainder of the simulation period.
        phy_params_dict = self._descale_phy_parameters(
            phy_params[warm_up:, :, :],
            dy_list=self.dynamic_params,
        )
        fluxes, states = self._PBM(x[warm_up:, :, :], current_states, phy_params_dict)

        # State caching
        self._states_cache = [s.detach() for s in states]

        if self.cache_states:
            self.states = self._states_cache

        return fluxes

    def _PBM(
        self,
        forcing: torch.Tensor,
        states: tuple,
        full_param_dict: dict,
    ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Run through process-based model (PBM).

        Parameters
        ----------
        forcing
            Input forcing data.
        states
            Initial model states.
        full_param_dict
            Dictionary of model parameters.

        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states

        # Forcings
        P = forcing[:, :, self.variables.index('prcp')]  # Precipitation
        T = forcing[:, :, self.variables.index('tmean')]  # Mean air temp
        PET = forcing[:, :, self.variables.index('pet')]  # Potential ET
        nsteps, ngrid = P.shape

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(nsteps, 1) * P

        # Initialize time series of model variables in shape [time, basins, nmul].
        Qsimmu = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.001
        Q0_sim = (
            torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        )
        Q1_sim = (
            torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        )
        Q2_sim = (
            torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        )

        AET = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        recharge_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        excs_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        evapfactor_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        tosoil_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        PERC_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SWE_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        capillary_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        param_dict = {}
        for t in range(nsteps):
            # Get dynamic parameter values per timestep.
            for key in full_param_dict.keys():
                param_dict[key] = full_param_dict[key][t, :, :]

            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(
                PRECIP, (Tm[t, :, :] >= param_dict['parTT']).type(torch.float32)
            )
            SNOW = torch.mul(
                PRECIP, (Tm[t, :, :] < param_dict['parTT']).type(torch.float32)
            )

            # Snow -------------------------------
            SNOWPACK = SNOWPACK + SNOW
            melt = param_dict['parCFMAX'] * (Tm[t, :, :] - param_dict['parTT'])
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = (
                param_dict['parCFR']
                * param_dict['parCFMAX']
                * (param_dict['parTT'] - Tm[t, :, :])
            )
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (param_dict['parCWH'] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation -------------------------------
            soil_wetness = (SM / param_dict['parFC']) ** param_dict['parBETA']
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge

            excess = SM - param_dict['parFC']
            excess = torch.clamp(excess, min=0.0)
            SM = SM - excess
            # NOTE: Different from HBV 1.0. Add static/dynamicET shape parameter parBETAET.
            evapfactor = (
                SM / (param_dict['parLP'] * param_dict['parFC'])
            ) ** param_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=self.nearzero)

            # Capillary rise (HBV 1.1p mod) -------------------------------
            capillary = torch.min(
                SLZ,
                param_dict['parC']
                * SLZ
                * (1.0 - torch.clamp(SM / param_dict['parFC'], max=1.0)),
            )

            SM = torch.clamp(SM + capillary, min=self.nearzero)
            SLZ = torch.clamp(SLZ - capillary, min=self.nearzero)

            # Groundwater boxes -------------------------------
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, param_dict['parPERC'])
            SUZ = SUZ - PERC
            Q0 = param_dict['parK0'] * torch.clamp(SUZ - param_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0
            Q1 = param_dict['parK1'] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = param_dict['parK2'] * SLZ
            SLZ = SLZ - Q2

            Qsimmu[t, :, :] = Q0 + Q1 + Q2
            Q0_sim[t, :, :] = Q0
            Q1_sim[t, :, :] = Q1
            Q2_sim[t, :, :] = Q2
            AET[t, :, :] = ETact
            SWE_sim[t, :, :] = SNOWPACK
            capillary_sim[t, :, :] = capillary

            recharge_sim[t, :, :] = recharge
            excs_sim[t, :, :] = excess
            evapfactor_sim[t, :, :] = evapfactor
            tosoil_sim[t, :, :] = tosoil
            PERC_sim[t, :, :] = PERC

        # Get the average or weighted average using learned weights.
        if self.muwts is None:
            Qsimavg = Qsimmu.mean(-1)
        else:
            Qsimavg = (Qsimmu * self.muwts).sum(-1)

        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if self.comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(nsteps, ngrid * self.nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            UH = uh_gamma(
                self.routing_param_dict['route_a'].repeat(nsteps, 1).unsqueeze(-1),
                self.routing_param_dict['route_b'].repeat(nsteps, 1).unsqueeze(-1),
                lenF=15,
            )
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])  # [gages,vars,time]
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = uh_conv(rf, UH).permute([2, 0, 1])

            # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time].
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q0_rout = uh_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q1_rout = uh_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q2_rout = uh_conv(rf_Q2, UH).permute([2, 0, 1])

            if self.comprout:
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(nsteps, ngrid, self.nmul)
                if self.muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * self.muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:
            # No routing, only output the average of all model sims.
            Qs = torch.unsqueeze(Qsimavg, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        states = (SNOWPACK, MELTWATER, SM, SUZ, SLZ)

        if self.initialize:
            # If initialize is True, only return warmed-up storages.
            return states
        else:
            # Baseflow index (BFI) calculation
            BFI_sim = (
                100
                * (torch.sum(Q2_rout, dim=0) / (torch.sum(Qs, dim=0) + self.nearzero))[
                    :, 0
                ]
            )

            # Return all sim results.
            flux_dict = {
                'streamflow': Qs,  # Routed Streamflow
                'srflow': Q0_rout,  # Routed surface runoff
                'ssflow': Q1_rout,  # Routed subsurface flow
                'gwflow': Q2_rout,  # Routed groundwater flow
                'AET_hydro': AET.mean(-1, keepdim=True),  # Actual ET
                'PET_hydro': PETm.mean(-1, keepdim=True),  # Potential ET
                'SWE': SWE_sim.mean(-1, keepdim=True),  # Snow water equivalent
                'streamflow_no_rout': Qsim.unsqueeze(dim=2),  # Streamflow
                'srflow_no_rout': Q0_sim.mean(-1, keepdim=True),  # Surface runoff
                'ssflow_no_rout': Q1_sim.mean(-1, keepdim=True),  # Subsurface flow
                'gwflow_no_rout': Q2_sim.mean(-1, keepdim=True),  # Groundwater flow
                'recharge': recharge_sim.mean(-1, keepdim=True),  # Recharge
                'excs': excs_sim.mean(-1, keepdim=True),  # Excess stored water
                'evapfactor': evapfactor_sim.mean(
                    -1, keepdim=True
                ),  # Evaporation factor
                'tosoil': tosoil_sim.mean(-1, keepdim=True),  # Infiltration
                'percolation': PERC_sim.mean(-1, keepdim=True),  # Percolation
                'capillary': capillary_sim.mean(-1, keepdim=True),  # Capillary rise
                'BFI': BFI_sim,  # Baseflow index
            }

            if not self.warm_up_states:
                for key in flux_dict.keys():
                    if key != 'BFI':
                        flux_dict[key] = flux_dict[key][self.pred_cutoff :, :, :]
            return flux_dict, states
