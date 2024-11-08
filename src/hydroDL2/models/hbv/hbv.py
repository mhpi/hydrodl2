import torch

from hydroDL2.core.calc import change_param_range
from hydroDL2.core.calc.uh_routing import UH_conv, UH_gamma



class HBVMulTDET(torch.nn.Module):
    """Multi-component Pytorch HBV model.

    Adapted from Farshid Rahmani, Yalan Song.

    Original NumPy version from Beck et al., 2020 (http://www.gloh2o.org/hbv/),
    which runs the HBV-light hydrological model (Seibert, 2005).
    """
    def __init__(self, config=None):
        super(HBVMulTDET, self).__init__()
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.static_idx = self.warm_up - 1
        self.dy_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = False
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.parameter_bounds = dict(
            parBETA=[1.0, 6.0],
            parFC=[50, 1000],
            parK0=[0.05, 0.9],
            parK1=[0.01, 0.5],
            parK2=[0.001, 0.2],
            parLP=[0.2, 1],
            parPERC=[0, 10],
            parUZL=[0, 100],
            parTT=[-2.5, 2.5],
            parCFMAX=[0.5, 10],
            parCFR=[0, 0.1],
            parCWH=[0, 0.2]
        )
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]

        if config is not None:
            # Overwrite defaults with config values.
            self.warm_up = config['phy_model']['warm_up']
            self.static_idx = config['phy_model']['stat_param_idx']
            self.dy_drop = config['dy_drop']
            self.dy_params = config['phy_model']['dy_params']['HBV']
            self.variables = config['phy_model']['forcings']
            self.routing = config['phy_model']['routing']
            self.nearzero = config['phy_model']['nearzero']
            self.nmul = config['nmul']

            if 'parBETAET' in config['phy_model']['dy_params']['HBV']:
                self.parameter_bounds['parBETAET'] = [0.3, 5]
                
    def forward(self, x, parameters, routing_parameters=None, muwts=None,
                comprout=False):
        """Forward pass for the HBV"""
        # Initialization
        if self.warm_up > 0:
            with torch.no_grad():
                xinit = x[0:self.warm_up, :, :]
                init_model = HBVMulTDET(self.config).to(self.device)

                # Defaults for warm-up.
                init_model.initialize = True
                init_model.warm_up = 0
                init_model.static_idx = self.warm_up-1
                init_model.muwts = None
                init_model.routing = False
                init_model.comprout = False
                init_model.dy_params = []

                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = init_model(
                    xinit,
                    parameters,
                    routing_parameters,
                    muwts=None,
                    comprout=False
                )
        else:
            # Without warm-up, initialize state variables with zeros.
            Ngrid = x.shape[1]
            SNOWPACK = (torch.zeros([Ngrid, self.nmul], dtype=torch.float32) + 0.001).to(self.device)
            MELTWATER = (torch.zeros([Ngrid, self.nmul], dtype=torch.float32) + 0.001).to(self.device)
            SM = (torch.zeros([Ngrid, self.nmul], dtype=torch.float32) + 0.001).to(self.device)
            SUZ = (torch.zeros([Ngrid, self.nmul], dtype=torch.float32) + 0.001).to(self.device)
            SLZ = (torch.zeros([Ngrid, self.nmul], dtype=torch.float32) + 0.001).to(self.device)

        # Parameters
        params_dict_raw = dict()
        for num, param in enumerate(self.parameter_bounds.keys()):
            params_dict_raw[param] = change_param_range(
                param=parameters[:, :, num, :],
                bounds=self.parameter_bounds[param]
            )

        # Forcings
        P = x[self.warm_up:, :, self.variables.index('prcp')]  # Precipitation
        T = x[self.warm_up:, :, self.variables.index('tmean')]  # Mean air temp
        PET = x[self.warm_up:, :, self.variables.index('pet')] # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, self.nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, self.nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, self.nmul)

        Nstep, Ngrid = P.size()

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(Nstep, 1) * P

        # Initialize time series of model variables in shape [time, basins, nmul].
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).to(self.device)
        Q0_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(self.device)
        Q1_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(self.device)
        Q2_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(self.device)

        AET = (torch.zeros(Pm.size(), dtype=torch.float32)).to(self.device)
        recharge_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(self.device)
        excs_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(self.device)
        evapfactor_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(self.device)
        tosoil_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(self.device)
        PERC_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(self.device)
        SWE_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(self.device)

        # Init static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in self.dy_params: # and len(params_raw.shape) > 2:
                # Use the last day of data as static parameter's value.
                params_dict[key] = params_dict_raw[key][self.static_idx, :, :]

        # Init dynamic parameters
        # (Use a dydrop ratio: fix a probability mask for setting dynamic params
        # as static in some basins.)
        if len(self.dy_params) > 0:
            params_dict_raw_dy = dict()
            pmat = torch.ones([Ngrid, 1]) * self.dy_drop
            for i, key in enumerate(self.dy_params):
                drmask = torch.bernoulli(pmat).detach_().to(self.device)
                dynPar = params_dict_raw[key]
                staPar = params_dict_raw[key][self.static_idx, :, :].unsqueeze(0).repeat([dynPar.shape[0], 1, 1])
                params_dict_raw_dy[key] = dynPar * (1 - drmask) + staPar * drmask

        for t in range(Nstep):
            # Get dynamic parameter values per timestep.
            for key in self.dy_params:
                params_dict[key] = params_dict_raw_dy[key][self.warm_up + t, :, :]

            # Separate precipitation into liquid and solid components.
            PRECIP = Pm[t, :, :]
            RAIN = torch.mul(PRECIP, (Tm[t, :, :] >= params_dict['parTT']).type(torch.float32))
            SNOW = torch.mul(PRECIP, (Tm[t, :, :] < params_dict['parTT']).type(torch.float32))

            # Snow -------------------------------
            SNOWPACK = SNOWPACK + SNOW
            melt = params_dict['parCFMAX'] * (Tm[t, :, :] - params_dict['parTT'])
            # melt[melt < 0.0] = 0.0
            melt = torch.clamp(melt, min=0.0)
            # melt[melt > SNOWPACK] = SNOWPACK[melt > SNOWPACK]
            melt = torch.min(melt, SNOWPACK)
            MELTWATER = MELTWATER + melt
            SNOWPACK = SNOWPACK - melt
            refreezing = params_dict['parCFR'] * params_dict['parCFMAX'] * (
                params_dict['parTT'] - Tm[t, :, :]
                )
            # refreezing[refreezing < 0.0] = 0.0
            # refreezing[refreezing > MELTWATER] = MELTWATER[refreezing > MELTWATER]
            refreezing = torch.clamp(refreezing, min=0.0)
            refreezing = torch.min(refreezing, MELTWATER)
            SNOWPACK = SNOWPACK + refreezing
            MELTWATER = MELTWATER - refreezing
            tosoil = MELTWATER - (params_dict['parCWH'] * SNOWPACK)
            tosoil = torch.clamp(tosoil, min=0.0)
            MELTWATER = MELTWATER - tosoil

            # Soil and evaporation -------------------------------
            soil_wetness = (SM / params_dict['parFC']) ** params_dict['parBETA']
            # soil_wetness[soil_wetness < 0.0] = 0.0
            # soil_wetness[soil_wetness > 1.0] = 1.0
            soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
            recharge = (RAIN + tosoil) * soil_wetness

            SM = SM + RAIN + tosoil - recharge

            excess = SM - params_dict['parFC']
            excess = torch.clamp(excess, min=0.0)
            # parBETAET only has effect when it is a dynamic parameter (=1 otherwise).
            evapfactor = (SM / (params_dict['parLP'] * params_dict['parFC']))
            if 'parBETAET' in params_dict:
                evapfactor = evapfactor ** params_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=self.nearzero)  # SM != 0 for grad tracking.

            # Groundwater boxes -------------------------------
            SUZ = SUZ + recharge + excess
            PERC = torch.min(SUZ, params_dict['parPERC'])
            SUZ = SUZ - PERC
            Q0 = params_dict['parK0'] * torch.clamp(SUZ - params_dict['parUZL'], min=0.0)
            SUZ = SUZ - Q0
            Q1 = params_dict['parK1'] * SUZ
            SUZ = SUZ - Q1
            SLZ = SLZ + PERC
            Q2 = params_dict['parK2'] * SLZ
            SLZ = SLZ - Q2

            Qsimmu[t, :, :] = Q0 + Q1 + Q2
            Q0_sim[t, :, :] = Q0
            Q1_sim[t, :, :] = Q1
            Q2_sim[t, :, :] = Q2
            AET[t, :, :] = ETact
            SWE_sim[t, :, :] = SNOWPACK

            recharge_sim[t, :, :] = recharge
            excs_sim[t, :, :] = excess
            evapfactor_sim[t, :, :] = evapfactor
            tosoil_sim[t, :, :] = tosoil
            PERC_sim[t, :, :] = PERC

        # Get the overall average 
        # or weighted average using learned weights.
        if muwts is None:
            Qsimavg = Qsimmu.mean(-1)
        else:
            Qsimavg = (Qsimmu * muwts).sum(-1)

        # Run routing
        if self.routing:
            # Routing for all components or just the average.
            if comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(Nstep, Ngrid * self.nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            # Scale routing params.
            temp_a = change_param_range(
                param=routing_parameters[:, 0],
                bounds=self.conv_routing_hydro_model_bound[0]
            )
            temp_b = change_param_range(
                param=routing_parameters[:, 1],
                bounds=self.conv_routing_hydro_model_bound[1]
            )
            rout_a = temp_a.repeat(Nstep, 1).unsqueeze(-1)
            rout_b = temp_b.repeat(Nstep, 1).unsqueeze(-1)

            UH = UH_gamma(rout_a, rout_b, lenF=15)  # lenF: folter
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

            if comprout: 
                # Qs is now shape [time, [gages*num models], vars]
                Qstemp = Qsrout.view(Nstep, Ngrid, self.nmul)
                if muwts is None:
                    Qs = Qstemp.mean(-1, keepdim=True)
                else:
                    Qs = (Qstemp * muwts).sum(-1, keepdim=True)
            else:
                Qs = Qsrout

        else:
            # No routing, only output the average of all model sims.
            Qs = torch.unsqueeze(Qsimavg, -1)
            Q0_rout = Q1_rout = Q2_rout = None

        if self.initialize:
            # Means we are in warm up. here we just return the storages to be
            # used as initial values. Only return model states for warmup.
            return Qs, SNOWPACK, MELTWATER, SM, SUZ, SLZ
        else:
            # Return all sim results.
            return dict(flow_sim=Qs,
                        srflow=Q0_rout,
                        ssflow=Q1_rout,
                        gwflow=Q2_rout,
                        AET_hydro=AET.mean(-1, keepdim=True),
                        PET_hydro=PETm.mean(-1, keepdim=True),
                        SWE=SWE_sim.mean(-1, keepdim=True),
                        flow_sim_no_rout=Qsim.unsqueeze(dim=2),
                        srflow_no_rout=Q0_sim.mean(-1, keepdim=True),
                        ssflow_no_rout=Q1_sim.mean(-1, keepdim=True),
                        gwflow_no_rout=Q2_sim.mean(-1, keepdim=True),
                        recharge=recharge_sim.mean(-1, keepdim=True),
                        excs=excs_sim.mean(-1, keepdim=True),
                        evapfactor=evapfactor_sim.mean(-1, keepdim=True),
                        tosoil=tosoil_sim.mean(-1, keepdim=True),
                        percolation=PERC_sim.mean(-1, keepdim=True),
                        )
