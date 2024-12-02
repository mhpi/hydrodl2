from typing import Any, Dict, Optional, Tuple, Union

import torch

from hydroDL2.core.calc import change_param_range
from hydroDL2.core.calc.uh_routing import UH_conv, UH_gamma


class PRMS(torch.nn.Module):
    """Multi-component Pytorch PRMS model.

    Adapted from Farshid Rahmani.

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
            'tt': [-3, 5],    # tt, Temperature threshold for snowfall and melt [oC]
            'ddf': [0, 20],    # ddf,  Degree-day factor for snowmelt [mm/oC/d]
            'alpha': [0, 1],     # alpha, Fraction of rainfall on soil moisture going to interception [-]
            'beta': [0, 1],    # beta, Fraction of catchment where rain goes to soil moisture [-]
            'stor': [0, 5],    # stor, Maximum interception capcity [mm]
            'retip': [0, 50],    # retip, Maximum impervious area storage [mm]
            'fscn': [0, 1],    # fscn, Fraction of SCX where SCN is located [-]
            'scx': [0, 1],    # scx, Maximum contributing fraction area to saturation excess flow [-]
            'flz': [0.005, 0.995],    # flz, Fraction of total soil moisture that is the lower zone [-]
            'stot': [1, 2000],    # stot, Total soil moisture storage [mm]: REMX+SMAX
            'cgw': [0, 20],    # cgw, Constant drainage to deep groundwater [mm/d]
            'resmax': [1, 300],    # resmax, Maximum flow routing reservoir storage (used for scaling only, there is no overflow) [mm]
            'k1': [0, 1],    # k1, Groundwater drainage coefficient [d-1]
            'k2': [1, 5],    # k2, Groundwater drainage non-linearity [-]
            'k3': [0, 1],    # k3, Interflow coefficient 1 [d-1]
            'k4': [0, 1],    # k4, Interflow coefficient 2 [mm-1 d-1]
            'k5': [0, 1],    # k5, Baseflow coefficient [d-1]
            'k6': [0, 1],    # k6, Groundwater sink coefficient [d-1],
        }
        self.routing_parameter_bounds = {
            'rout_a': [0, 2.9],
            'rout_b': [0, 6.5]
        }

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config.get('warm_up', self.warm_up)
            self.warm_up_states = config.get('warm_up_states', self.warm_up_states)
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get('PRMS', self.dynamic_params)
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

        self.learnable_param_count = len(self.phy_param_names) * self.nmul \
            + len(self.routing_param_names)

    def unpack_parameters(
            self,
            parameters: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        # Physical parameters
        phy_params = torch.sigmoid(
            parameters[:, :, :phy_param_count * self.nmul]).view(
                parameters.shape[0],
                parameters.shape[1],
                phy_param_count,
                self.nmul
            )
        # Routing parameters
        routing_params = None
        if self.routing == True:
            routing_params = torch.sigmoid(
                parameters[-1, :, phy_param_count * self.nmul:]
            )
        return phy_params, routing_params

    def descale_phy_parameters(
            self,
            phy_params: torch.Tensor,
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
        n_steps = phy_params.size(0)
        n_grid = phy_params.size(1)

        param_dict = {}
        pmat = torch.ones([1, n_grid, 1]) * self.dy_drop
        for i, name in enumerate(self.parameter_bounds.keys()):
            staPar = phy_params[-1, :, i,:].unsqueeze(0).repeat([n_steps, 1, 1])
            if name in dy_list:
                dynPar = phy_params[:, :, i,:]
                drmask = torch.bernoulli(pmat).detach_().cuda() 
                comPar = dynPar * (1 - drmask) + staPar * drmask
                param_dict[name] = change_param_range(
                    param=comPar,
                    bounds=self.parameter_bounds[name]
                )
            else:
                param_dict[name] = change_param_range(
                    param=staPar,
                    bounds=self.parameter_bounds[name]
                )
        return param_dict

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
        """Forward pass for PRMS.
        
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
        self.muwts = x_dict.get('muwts', None)

        # Unpack parameters.
        phy_params, routing_params = self.unpack_parameters(parameters)
        
        if self.routing:
            self.routing_param_dict = self.descale_rout_parameters(routing_params)

        # Initialization
        if self.warm_up_states:
            warm_up = self.warm_up
        else:
            # No state warm up - run the full model for warm_up days.
            self.pred_cutoff = self.warm_up
            warm_up = 0
        
        n_grid = x.size(1)

        # Initialize model states.
        # snow storage
        snow_storage = torch.zeros([n_grid, self.nmul],
                                   dtype=torch.float32,
                                   device=self.device) + 0.001
        # interception storage
        XIN_storage = torch.zeros([n_grid, self.nmul],
                                  dtype=torch.float32,
                                  device=self.device) + 0.001
        # RSTOR storage
        RSTOR_storage = torch.zeros([n_grid, self.nmul],
                                    dtype=torch.float32,
                                    device=self.device) + 0.001
        # storage in upper soil moisture zone
        RECHR_storage = torch.zeros([n_grid, self.nmul],
                                    dtype=torch.float32,
                                    device=self.device) + 0.001
        # storage in lower soil moisture zone
        SMAV_storage = torch.zeros([n_grid, self.nmul],
                                   dtype=torch.float32,
                                   device=self.device) + 0.001
        # storage in runoff reservoir
        RES_storage = torch.zeros([n_grid, self.nmul],
                                  dtype=torch.float32,
                                  device=self.device) + 0.001
        # GW storage
        GW_storage = torch.zeros([n_grid, self.nmul],
                                 dtype=torch.float32,
                                 device=self.device) + 0.001
        
        # Warm-up model states - run the model only on warm_up days first.
        if warm_up > 0:
            with torch.no_grad():
                phy_param_warmup_dict = self.descale_phy_parameters(
                    phy_params[:warm_up,:,:],
                    dy_list=[]
                )
                # Save current model settings.
                initialize = self.initialize
                routing  = self.routing

                # Set model settings for warm-up.
                self.initialize =  True
                self.routing = False

                snow_storage, XIN_storage, RSTOR_storage, RECHR_storage, \
                    SMAV_storage, RES_storage, GW_storage = self.PBM(
                        x[:warm_up, :, :],
                        [snow_storage, XIN_storage, RSTOR_storage, RECHR_storage,
                         SMAV_storage, RES_storage, GW_storage],
                        phy_param_warmup_dict
                )

                # Restore model settings.
                self.initialize = initialize
                self.routing = routing
        
        phy_params_dict = self.descale_phy_parameters(
            phy_params[warm_up:,:,:],
            dy_list=self.dynamic_params
        )
        
        # Run the model for the remainder of simulation period.
        return self.PBM(
                    x[warm_up:, :, :],
                    [snow_storage, XIN_storage, RSTOR_storage, RECHR_storage,
                     SMAV_storage, RES_storage, GW_storage],
                    phy_params_dict
                )

    def PBM(
            self,
            forcing: torch.Tensor,
            states: Tuple,
            full_param_dict: Dict
        ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Run the PRMS model forward.
        
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
        snow_storage, XIN_storage, RSTOR_storage, RECHR_storage, SMAV_storage, \
            RES_storage, GW_storage = states

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
        Q_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        sas_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        sro_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        bas_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        ras_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        snk_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        # AET = PET_coef * PET
        AET = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        inf_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        PC_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SEP_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        GAD_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        ea_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        qres_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        
        param_dict ={}
        for t in range(n_steps):
            # Get dynamic parameter values per timestep.
            for key in full_param_dict.keys():
                param_dict[key] = full_param_dict[key][t, :, :]

            scn = param_dict['fscn'] * param_dict['scx']
            remx = (1 - param_dict['flz']) * param_dict['stot']
            smax = param_dict['flz'] * param_dict['stot']

            delta_t = 1  # timestep (day)
            PRECIP = Pm[t, :, :]
            Ep = PETm[t, :, :]
            temp = Tm[t, :, :]

            # Fluxes
            flux_ps = torch.mul(PRECIP, (temp <= param_dict['tt']).type(torch.float32))
            flux_pr = torch.mul(PRECIP, (temp > param_dict['tt']).type(torch.float32))
            snow_storage = snow_storage + flux_ps
            flux_m = torch.clamp(param_dict['ddf'] * (temp - param_dict['tt']), min=0.0)
            flux_m = torch.min(flux_m, snow_storage/delta_t)
            # flux_m = torch.clamp(flux_m, min=0.0)
            snow_storage = torch.clamp(snow_storage - flux_m, min=self.nearzero)

            flux_pim = flux_pr * (1 - param_dict['beta'])
            flux_psm = flux_pr * param_dict['beta']
            flux_pby = flux_psm * (1 - param_dict['alpha'])
            flux_pin = flux_psm * param_dict['alpha']

            XIN_storage = XIN_storage + flux_pin
            flux_ptf = XIN_storage - param_dict['stor']
            flux_ptf = torch.clamp(flux_ptf, min=0.0)
            XIN_storage = torch.clamp(XIN_storage - flux_ptf, min=self.nearzero)
            evap_max_in = Ep * param_dict['beta']   # only can happen in pervious area
            flux_ein = torch.min(evap_max_in, XIN_storage/delta_t)
            XIN_storage = torch.clamp(XIN_storage - flux_ein, min=self.nearzero)

            flux_mim = flux_m * (1 - param_dict['beta'])
            flux_msm = flux_m * param_dict['beta']
            RSTOR_storage = RSTOR_storage + flux_mim + flux_pim
            flux_sas = RSTOR_storage - param_dict['retip']
            flux_sas = torch.clamp(flux_sas, min=0.0)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_sas, min=self.nearzero)
            evap_max_im = (1 - param_dict['beta']) * Ep
            flux_eim = torch.min(evap_max_im, RSTOR_storage / delta_t)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_eim, min=self.nearzero)

            sro_lin_ratio = scn + (param_dict['scx'] - scn) * (RECHR_storage / remx)
            sro_lin_ratio = torch.clamp(sro_lin_ratio, min=0.0, max=1.0)
            flux_sro = sro_lin_ratio * (flux_msm + flux_ptf + flux_pby)
            flux_inf = flux_msm + flux_ptf + flux_pby - flux_sro
            RECHR_storage = RECHR_storage + flux_inf
            flux_pc = RECHR_storage - remx
            flux_pc = torch.clamp(flux_pc, min=0.0)
            RECHR_storage = RECHR_storage - flux_pc
            evap_max_a = (RECHR_storage / remx) * (Ep - flux_ein - flux_eim)
            evap_max_a = torch.clamp(evap_max_a, min=0.0)
            flux_ea = torch.min(evap_max_a, RECHR_storage / delta_t)
            RECHR_storage = torch.clamp(RECHR_storage - flux_ea, min=self.nearzero)

            SMAV_storage = SMAV_storage + flux_pc
            flux_excs = SMAV_storage - smax
            flux_excs = torch.clamp(flux_excs, min=0.0)
            SMAV_storage = SMAV_storage - flux_excs
            transp = torch.where(RECHR_storage < (Ep - flux_ein - flux_eim),
                                 (SMAV_storage/smax) * (Ep - flux_ein - flux_eim - flux_ea),
                                 torch.zeros(flux_excs.shape, dtype=torch.float32, device=self.device))
            transp = torch.clamp(transp, min=0.0)    # in case Ep - flux_ein - flux_eim - flux_ea was negative
            SMAV_storage = torch.clamp(SMAV_storage - transp, min=self.nearzero)

            flux_sep = torch.min(param_dict['cgw'], flux_excs)
            flux_qres = torch.clamp(flux_excs - flux_sep, min=0.0)

            RES_storage = RES_storage + flux_qres
            flux_gad = param_dict['k1'] * ((RES_storage / param_dict['resmax']) ** param_dict['k2'])
            flux_gad = torch.min(flux_gad, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_gad, min=self.nearzero)
            flux_ras = param_dict['k3'] * RES_storage + param_dict['k4'] * (RES_storage ** 2)
            flux_ras = torch.min(flux_ras, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_ras, min=self.nearzero)
            # RES_excess = RES_storage - resmax[:, t, :]   # if there is still overflow, it happend in discrete version
            # RES_excess = torch.clamp(RES_excess, min=0.0)
            # flux_ras = flux_ras + RES_excess
            # RES_storage = torch.clamp(RES_storage - RES_excess, min=self.nearzero)


            GW_storage = GW_storage + flux_gad + flux_sep
            flux_bas = param_dict['k5'] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_bas, min=self.nearzero)
            flux_snk = param_dict['k6'] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_snk, min=self.nearzero)

            Q_sim[t, :, :] = (flux_sas + flux_sro + flux_bas + flux_ras)
            sas_sim[t, :, :] = flux_sas
            sro_sim[t, :, :] = flux_sro
            bas_sim[t, :, :] = flux_bas
            ras_sim[t, :, :] = flux_ras
            snk_sim[t, :, :] = flux_snk
            AET[t, :, :] = flux_ein + flux_eim + flux_ea + transp
            inf_sim[t, :, :] = flux_inf
            PC_sim[t, :, :] = flux_pc
            SEP_sim[t, :, :] = flux_sep
            GAD_sim[t, :, :] = flux_gad
            ea_sim[t, :, :] = flux_ea
            qres_sim[t, :, :] = flux_qres

        # Get the overall average 
        # or weighted average using learned weights.
        if self.muwts is None:
            Qsimavg = Q_sim.mean(-1)
        else:
            Qsimavg = (Q_sim * self.muwts).sum(-1)
        
        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if self.comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Q_sim.view(n_steps, n_grid * self.nmul)
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

            # Routing individual layers, all w/ dims [gages,vars,time].
            rf_sas = sas_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qsas_rout = UH_conv(rf_sas, UH).permute([2, 0, 1])

            rf_sro = sro_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qsro_rout = UH_conv(rf_sro, UH).permute([2, 0, 1])

            rf_ras = ras_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qras_rout = UH_conv(rf_ras, UH).permute([2, 0, 1])

            rf_bas = bas_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qbas_rout = UH_conv(rf_bas, UH).permute([2, 0, 1])

            if self.comprout: 
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(n_steps, n_grid, self.nmul)
                if self.muwts is None:
                    Qsrout = Qstemp.mean(-1, keepdim=True)
                else:
                    Qsrout = (Qstemp *self.muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout
            
        else:
            # No routing, only output the average of all model sims.
            Qsrout = torch.unsqueeze(Qsimavg, -1)
            Qsas_rout = Qsro_rout = Qbas_rout = Qras_rout = None

        if self.initialize:
            # If initialize is True, only return warmed-up storages.
            return snow_storage, XIN_storage, RSTOR_storage, RECHR_storage, \
                SMAV_storage, RES_storage, GW_storage
        else:
            # Baseflow index (BFI) calculation
            BFI_sim = 100 * (torch.sum(Qbas_rout, dim=0) / (
                torch.sum(Qsrout, dim=0) + self.nearzero))[:,0]
            
            # Return all sim results.
            out_dict = {
                'flow_sim': Qsrout,
                'srflow': Qsas_rout + Qsro_rout,
                'ssflow': Qras_rout,
                'gwflow': Qbas_rout,
                'sink': torch.mean(snk_sim, -1).unsqueeze(-1),
                'PET_hydro': PETm.mean(-1, keepdim=True),
                'AET_hydro': AET.mean(-1, keepdim=True),
                'flow_sim_no_rout': Q_sim.mean(-1, keepdim=True),
                'srflow_no_rout': (sas_sim + sro_sim).mean(-1, keepdim=True),
                'ssflow_no_rout': ras_sim.mean(-1, keepdim=True),
                'gwflow_no_rout': bas_sim.mean(-1, keepdim=True),
                'flux_inf': inf_sim.mean(-1, keepdim=True),
                'flux_pc': PC_sim.mean(-1, keepdim=True),
                'flux_sep': SEP_sim.mean(-1, keepdim=True),
                'flux_gad': GAD_sim.mean(-1, keepdim=True),
                'flux_ea': ea_sim.mean(-1, keepdim=True),
                'flux_qres': qres_sim.mean(-1, keepdim=True),
                'BFI_sim': BFI_sim
            }
            
            if not self.warm_up_states:
                for key in out_dict.keys():
                    if key != 'BFI_sim':
                        out_dict[key] = out_dict[key][self.pred_cutoff:, :, :]
            return out_dict
