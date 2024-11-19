from typing import Dict, Tuple, Union

import torch

from hydroDL2.core.calc import change_param_range
from hydroDL2.core.calc.uh_routing import UH_conv, UH_gamma


class PRMS(torch.nn.Module):
    """Multi-component Pytorch PRMS model.

    Adapted from Farshid Rahmani.
    """
    def __init__(self, config=None, device=None):
        super().__init__()
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.static_idx = self.warm_up - 1
        self.dy_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = False
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
            self.warm_up = config['phy_model']['warm_up']
            self.static_idx = config['phy_model']['stat_param_idx']
            self.dy_drop = config['dy_drop']
            self.dy_params = config['phy_model']['dy_params']['PRMS']
            self.variables = config['phy_model']['forcings']
            self.routing = config['phy_model']['routing']
            self.nearzero = config['phy_model']['nearzero']
            self.nmul = config['nmul']

        self.set_parameters()

    def set_parameters(self):
        """Get HBV model parameters."""
        phy_params = self.parameter_bounds.keys()
        if self.routing == True:
            rout_params = self.routing_parameter_bounds.keys()
        else:
            rout_params = []
        
        self.all_parameters = list(phy_params) + list(rout_params)
        self.learnable_param_count = len(phy_params) * self.nmul + len(rout_params)

    def unpack_parameters(
            self,
            parameters: torch.Tensor,
            n_steps: int,
            n_grid: int
        ) -> Dict:
        """Extract physics model parameters from NN output.
        
        Parameters
        ----------
        parameters : torch.Tensor
            Unprocessed, learned parameters from a neural network.
        n_steps : int
            Number of time steps in the input data.
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
        if self.routing == True:
            routing_params = torch.sigmoid(
                parameters[-1, :, phy_param_count * self.nmul:]
            )

        # Precompute probability mask for dynamic parameters
        if len(self.dy_params) > 0:
            pmat = torch.ones([n_grid, 1]) * self.dy_drop

        parameter_dict = {}
        for i, name in enumerate(self.all_parameters):
            if i < phy_param_count:
                # Physical parameters
                param = phy_params[self.static_idx, :, i, :]

                if name in self.dy_params:
                    # Make the parameter dynamic
                    drmask = torch.bernoulli(pmat).detach_().to(self.device)
                    dynamic_param = phy_params[:, :, i, :]

                    # Allow chance for dynamic parameter to be static
                    static_param = param.unsqueeze(0).repeat([dynamic_param.shape[0], 1, 1])
                    param = dynamic_param * (1 - drmask) + static_param * drmask
                
                parameter_dict[name] = change_param_range(
                    param=param,
                    bounds=self.parameter_bounds[name]
                )
            elif self.routing:
                # Routing parameters
                parameter_dict[name] = change_param_range(
                    param=routing_params[:, i - phy_param_count],
                    bounds=self.routing_parameter_bounds[name]
                ).repeat(n_steps, 1).unsqueeze(-1)
            else:
                break
        return parameter_dict

    def forward(
            self,
            x_dict: Dict[str, torch.Tensor],
            parameters: torch.Tensor
        ) -> Union[Tuple, Dict[str, torch.Tensor]]:
        """Forward pass for PRMS."""
        # Unpack input data.
        x = x_dict['x_phy']

        # Initialization
        if self.warm_up > 0:
            with torch.no_grad():
                x_init = {'x_phy': x[0:self.warm_up, :, :]}
                init_model = PRMS(self.config, device=self.device)

                # Defaults for warm-up.
                init_model.initialize = True
                init_model.warm_up = 0
                init_model.static_idx = self.warm_up-1
                init_model.muwts = None
                init_model.routing = False
                init_model.dy_params = []

                Q_init, snow_storage, XIN_storage, RSTOR_storage, \
                    RECHR_storage, SMAV_storage, \
                    RES_storage, GW_storage = init_model(
                    x_init,
                    parameters
                )
        else:
            # Without warm-up, initialize state variables with zeros.
            n_grid = x.shape[1]

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

        # Forcings
        P = x[self.warm_up:, :, self.variables.index('prcp')]  # Precipitation
        T = x[self.warm_up:, :, self.variables.index('tmean')]  # Mean air temp
        PET = x[self.warm_up:, :, self.variables.index('pet')] # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        n_steps, n_grid = P.size()

        # Parameters
        full_param_dict = self.unpack_parameters(parameters, n_steps, n_grid)

        # AET = PET_coef * PET
        # Initialize time series of model variables in shape [time, basins, nmul].
        Q_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        sas_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        sro_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        bas_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        ras_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        snk_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        AET = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        inf_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        PC_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        SEP_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        GAD_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        ea_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)
        qres_sim = torch.zeros(Pm.size(), dtype=torch.float32, device=self.device)

        param_dict = full_param_dict.copy()
        for t in range(n_steps):
            # Get dynamic parameter values per timestep.
            for key in self.dy_params:
                param_dict[key] = full_param_dict[key][self.warm_up + t, :, :]

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

        if self.routing:
            UH = UH_gamma(param_dict['rout_a'], param_dict['rout_b'], lenF=15)
            rf = Q_sim.mean(-1, keepdim=True).permute([1, 2, 0])  # [gages,vars,time]
            UH = UH.permute([1, 2, 0])  # [gages,vars,time]
            Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

            rf_sas = sas_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qsas_rout = UH_conv(rf_sas, UH).permute([2, 0, 1])

            rf_sro = sro_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qsro_rout = UH_conv(rf_sro, UH).permute([2, 0, 1])

            rf_ras = ras_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qras_rout = UH_conv(rf_ras, UH).permute([2, 0, 1])

            rf_bas = bas_sim.mean(-1, keepdim=True).permute([1, 2, 0])
            Qbas_rout = UH_conv(rf_bas, UH).permute([2, 0, 1])

        else:
            Qsrout = Q_sim.mean(-1, keepdim=True)
            Qsas_rout = sas_sim.mean(-1, keepdim=True)
            Qsro_rout = sro_sim.mean(-1, keepdim=True)
            Qbas_rout = bas_sim.mean(-1, keepdim=True)
            Qras_rout = ras_sim.mean(-1, keepdim=True)

        if self.initialize:
            # If initialize is True, it is warm-up mode; only return storages (states).
            return Qsrout, snow_storage, XIN_storage, RSTOR_storage, \
                RECHR_storage, SMAV_storage, RES_storage, GW_storage
        else:
            # Baseflow index (BFI) calculation
            BFI_sim = 100 * (torch.sum(Qbas_rout, dim=0) / (
                torch.sum(Qsrout, dim=0) + self.nearzero))[:,0]
            
            return {
                'flow_sim': Qsrout,
                'srflow': Qsas_rout + Qsro_rout,
                'ssflow': Qras_rout,
                'gwflow': Qbas_rout,
                'sink': torch.mean(snk_sim, -1).unsqueeze(-1),
                'PET_hydro': PET.mean(-1, keepdim=True),
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
                'BFI_sim': BFI_sim.mean(-1, keepdim=True)   
            }
