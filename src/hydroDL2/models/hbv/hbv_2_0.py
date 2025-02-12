from typing import Any, Dict, Optional, Tuple, Union

import torch

from hydroDL2.core.calc import change_param_range
from hydroDL2.core.calc.uh_routing import UH_conv, UH_gamma


class HBVUnitBasin(torch.nn.Module):
    """
    Multi-component, multi-scale, differentiable PyTorch HBV model with rainfall
    runoff simulation on unit basins.

    Authors
    -------
    -   Yalan Song
    -   (Original NumPy HBV ver.) Beck et al., 2020 (http://www.gloh2o.org/hbv/).
    -   (HBV-light Version 2) Seibert, 2005 (https://www.geo.uzh.ch/dam/jcr:c8afa73c-ac90-478e-a8c7-929eed7b1b62/HBV_manual_2005.pdf).

    Publication
    -----------
    -   Yalan Song, Tadd Bindas, Chaopeng Shen, et al. High-resolution
        national-scale water modeling is enhanced by multiscale differentiable
        physics-informed machine learning. ESS Open Archive . September 26, 2024.
        https://essopenarchive.org/doi/full/10.22541/essoar.172736277.74497104

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.
    device : torch.device, optional
        Device to run the model on.
    """
    def __init__(
            self,
            config: Optional[Dict[str, Any]] = None,
            device: Optional[torch.device] = None
        ) -> None:
        super().__init__()
        self.name = 'HBV 2.0UH'
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
        self.device = device
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
            'parRT': [0, 20],
            'parAC': [0, 2500],

        }
        self.routing_parameter_bounds = {
            'rout_a': [0, 2.9],
            'rout_b': [0, 6.5],
        }

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get('HBV_2_0', self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.routing = config.get('routing', self.routing)
            self.comprout = config.get('comprout', self.comprout)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
        self.set_parameters()

    def set_parameters(self) -> None:
        """Get physical parameters."""
        self.phy_param_names = self.parameter_bounds.keys()
        if self.routing == True:
            self.routing_param_names = self.routing_parameter_bounds.keys()
        else:
            self.routing_param_names = []

        self.learnable_param_count1 = len(self.dynamic_params) * self.nmul 
        self.learnable_param_count2 = (len(self.phy_param_names) - len(self.dynamic_params)) * self.nmul \
            + len(self.routing_param_names)
        self.learnable_param_count = self.learnable_param_count1 + self.learnable_param_count2

    def unpack_parameters(
            self,
            parameters: torch.Tensor,
        ) -> Dict[str, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.
        
        Parameters
        ----------
        parameters : torch.Tensor
            Unprocessed, learned parameters from a neural network.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of physical and routing parameters.
        """
        phy_param_count = len(self.parameter_bounds)
        dy_param_count = len(self.dynamic_params)
        dif_count = phy_param_count - dy_param_count

        
        # Physical dynamic parameters
        phy_dy_params = parameters[0].view(
                parameters[0].shape[0],
                parameters[0].shape[1],
                dy_param_count,
                self.nmul,
            )
        
        # Physical static parameters
        phy_static_params =  parameters[1][:,  :dif_count * self.nmul].view(
            parameters[1].shape[0],
            dif_count,
            self.nmul,
        )
            
        # Routing parameters
        routing_params = None
        if self.routing == True:
            routing_params = parameters[1][:,  dif_count * self.nmul:]
            
        return phy_dy_params, phy_static_params, routing_params

    def descale_phy_dy_parameters(
            self,
            phy_dy_params: torch.Tensor,
            dy_list:list,
        ) -> torch.Tensor:
        """Descale physical parameters.
        
        Parameters
        ----------
        phy_params : torch.Tensor
            Normalized physical parameters.
        dy_list : list
            List of dynamic parameters.
        
        Returns
        -------
        dict
            Dictionary of descaled physical parameters.
        """
        n_steps = phy_dy_params.size(0)
        n_grid = phy_dy_params.size(1)

        # TODO: Fix; if dynamic parameters are not entered in config as they are
        # in HBV params list, then descaling misamtch will occur.
        param_dict = {}
        pmat = torch.ones([1, n_grid, 1]) * self.dy_drop
        for i, name in enumerate(dy_list):
            staPar = phy_dy_params[-1, :, i,:].unsqueeze(0).repeat([n_steps, 1, 1])
         
            dynPar = phy_dy_params[:, :, i,:]
            drmask = torch.bernoulli(pmat).detach_().cuda() 
            comPar = dynPar * (1 - drmask) + staPar * drmask
            param_dict[name] = change_param_range(
                param=comPar,
                bounds=self.parameter_bounds[name]
            )
        return param_dict

    def descale_phy_stat_parameters(
            self,
            phy_stat_params: torch.Tensor,
            stat_list:list,
        ) -> torch.Tensor:
        """Descale routing parameters.
        
        Parameters
        ----------
        routing_params : torch.Tensor
            Normalized routing parameters.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """
        parameter_dict = {}
        for i, name in enumerate(stat_list):
            param = phy_stat_params[:, i,:]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.parameter_bounds[name]
            )
        return parameter_dict

    def descale_rout_parameters(
            self,
            routing_params: torch.Tensor
        ) -> torch.Tensor:
        """Descale routing parameters.
        
        Parameters
        ----------
        routing_params : torch.Tensor
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
                bounds=self.routing_parameter_bounds[name]
            )
        return parameter_dict

    def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            parameters: torch.Tensor
        ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Forward pass for HBV1.1p.
        
        Parameters
        ----------
        x_dict : dict
            Dictionary of input forcing data.
        parameters : torch.Tensor
            Unprocessed, learned parameters from a neural network.
        
        Returns
        -------
        Union[Tuple, dict]
            Tuple or dictionary of model outputs.
        """
        # Unpack input data.
        x = x_dict['x_phy']
        Ac = x_dict['ac_all'].unsqueeze(-1).repeat(1, self.nmul)
        Elevation = x_dict['elev_all'].unsqueeze(-1).repeat(1, self.nmul)
        self.muwts = x_dict.get('muwts', None)

        # Unpack parameters.
        phy_dy_params, phy_static_params, routing_params = self.unpack_parameters(parameters)
        
        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(routing_params)

        
        n_grid = x.size(1)

        # Initialize model states.
        SNOWPACK = torch.zeros([n_grid, self.nmul],
                                dtype=torch.float32,
                                device=self.device) + 0.001
        MELTWATER = torch.zeros([n_grid, self.nmul],
                                dtype=torch.float32,
                                device=self.device) + 0.001
        SM = torch.zeros([n_grid, self.nmul],
                         dtype=torch.float32,
                         device=self.device) + 0.001
        SUZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001
        SLZ = torch.zeros([n_grid, self.nmul],
                          dtype=torch.float32,
                          device=self.device) + 0.001

        phy_dy_params_dict = self.descale_phy_dy_parameters(
            phy_dy_params,
            dy_list=self.dynamic_params
        )

        phy_static_params_dict = self.descale_phy_stat_parameters(
            phy_static_params,
            stat_list=[param for param in self.phy_param_names if param not in self.dynamic_params]
        )

        # Run the model for the remainder of simulation period.
        return self.PBM( 
                    x,
                    Ac,
                    Elevation,
                    [SNOWPACK, MELTWATER, SM, SUZ, SLZ],
                    phy_dy_params_dict,
                    phy_static_params_dict
                )

    def PBM(
            self,
            forcing: torch.Tensor,
            Ac:torch.Tensor,
            Elevation:torch.Tensor,
            states: Tuple,
            phy_dy_params_dict: Dict,
            phy_static_params_dict: Dict
        ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Run the HBV1.1p model forward.
        
        Parameters
        ----------
        forcing : torch.Tensor
            Input forcing data.
        states : Tuple
            Initial model states.
        full_param_dict : dict
            Dictionary of model parameters.
        
        Returns
        -------
        Union[Tuple, dict]
            Tuple or dictionary of model outputs.
        """
        SNOWPACK, MELTWATER, SM, SUZ, SLZ = states

        # Forcings
        P = forcing[:, :, self.variables.index('prcp')]  # Precipitation
        T = forcing[:, :, self.variables.index('tmean')]  # Mean air temp
        PET = forcing[:, :, self.variables.index('pet')] # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        n_steps, n_grid = P.size()

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(n_steps, 1) * P

        # Initialize time series of model variables in shape [time, basins, nmul].
        Qsimmu = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.001
        Q0_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        Q1_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001
        Q2_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device) + 0.0001

        AET = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        recharge_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        excs_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        evapfactor_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        tosoil_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        PERC_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SWE_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        capillary_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        
        param_dict ={}
        for t in range(n_steps):
            # Get dynamic parameter values per timestep.
            for key in phy_dy_params_dict.keys():
                param_dict[key] = phy_dy_params_dict[key][t, :, :]
            for key in phy_static_params_dict.keys():
                param_dict[key] = phy_static_params_dict[key][:, :]
            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :, :]
            parTT_new = (Elevation >= 2000).type(torch.float32)*4.0 + (Elevation < 2000).type(torch.float32)*param_dict['parTT']
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= parTT_new).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < parTT_new).type(torch.float32))

            # Snow -------------------------------
            SNOWPACK = SNOWPACK + SNOW
            melt = param_dict['parCFMAX'] * (Tm[t, :, :] - parTT_new)
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = param_dict['parCFR'] * param_dict['parCFMAX'] * (
                parTT_new - Tm[t, :, :]
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
            evapfactor = (SM / (param_dict['parLP'] * param_dict['parFC'])) ** param_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=self.nearzero)

            # Capillary rise (HBV 1.1p mod) -------------------------------
            capillary = torch.min(SLZ, param_dict['parC'] * SLZ * (1.0 - torch.clamp(SM / param_dict['parFC'], max=1.0)))

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

            LF = torch.clamp((Ac-param_dict['parAC'])/1000,min = -1, max = 1) * param_dict['parRT']*(Ac<2500)+\
            torch.exp(torch.clamp(-(Ac-2500)/50, min = -10.0,max = 0.0))* param_dict['parRT']*(Ac>=2500)
            SLZ = torch.clamp(SLZ + LF, min=0.0)

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

        # Get the overall average 
        # or weighted average using learned weights.
        if self.muwts is None:
            Qsimavg = Qsimmu.mean(-1)
        else:
            Qsimavg = (Qsimmu * self.muwts).sum(-1)

        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if self.comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(n_steps, n_grid * self.nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            UH = UH_gamma(
                self.routing_param_dict['rout_a'].repeat(n_steps, 1).unsqueeze(-1),
                self.routing_param_dict['rout_b'].repeat(n_steps, 1).unsqueeze(-1),
                lenF=15
            )
            rf = torch.unsqueeze(Qsim, -1).permute([1, 2, 0])  # [gages,vars,time]
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            # Routing individually for Q0, Q1, and Q2, all w/ dims [gages,vars,time].
            rf_Q0 = Q0_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q0_rout = UH_conv(rf_Q0, UH).permute([2, 0, 1])
            rf_Q1 = Q1_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q1_rout = UH_conv(rf_Q1, UH).permute([2, 0, 1])
            rf_Q2 = Q2_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Q2_rout = UH_conv(rf_Q2, UH).permute([2, 0, 1])

            if self.comprout: 
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(n_steps, n_grid, self.nmul)
                if self.muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp *self.muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:
            # No routing, only output the average of all model sims.
            Qs = torch.unsqueeze(Qsimavg, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        if self.initialize:
            # If initialize is True, only return warmed-up storages.
            return SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # Baseflow index (BFI) calculation
            BFI_sim = 100 * (torch.sum(Q2_rout, dim=0) / (
                torch.sum(Qs, dim=0) + self.nearzero))[:,0]

            # Return all sim results.
            out_dict = {
                'flow_sim': Qs,
                'srflow': Q0_rout,
                'ssflow': Q1_rout,
                'gwflow': Q2_rout,
                'AET_hydro': AET.mean(-1, keepdim=True),
                'PET_hydro': PETm.mean(-1, keepdim=True),
                'SWE': SWE_sim.mean(-1, keepdim=True),
                'flow_sim_no_rout': Qsim.unsqueeze(dim=2),
                'srflow_no_rout': Q0_sim.mean(-1, keepdim=True),
                'ssflow_no_rout': Q1_sim.mean(-1, keepdim=True),
                'gwflow_no_rout': Q2_sim.mean(-1, keepdim=True),
                'recharge': recharge_sim.mean(-1, keepdim=True),
                'excs': excs_sim.mean(-1, keepdim=True),
                'evapfactor': evapfactor_sim.mean(-1, keepdim=True),
                'tosoil': tosoil_sim.mean(-1, keepdim=True),
                'percolation': PERC_sim.mean(-1, keepdim=True),
                'capillary': capillary_sim.mean(-1, keepdim=True),
                'BFI_sim': BFI_sim,
            }
            
            if not self.warm_up_states:
                for key in out_dict.keys():
                    if key != 'BFI_sim':
                        out_dict[key] = out_dict[key][self.pred_cutoff:, :, :]
            return out_dict
