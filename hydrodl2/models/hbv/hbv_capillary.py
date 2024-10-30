import torch
from models.pet_models.potet import get_potet
from core.calc.hydrograph import UH_gamma, UH_conv
from core.utils.utils import change_param_range, param_bounds_2D

import torch.nn.functional as F


class HBVMulTDET(torch.nn.Module):
    """
    Multi-component HBV Model Pytorch version (dynamic and static param capable)
    with capillary rise mod, adapted from dPL_Hydro_SNTEMP @ Farshid Rahmani.
    
    Supports optional Evapotranspiration parameter ET.

    Modified from the original numpy version from Beck et al., 2020
    (http://www.gloh2o.org/hbv/), which runs the HBV-light hydrological model
    (Seibert, 2005).
    """
    def __init__(self):
        super(HBVMulTDET, self).__init__()
        self.parameters_bound = dict(parBETA=[1.0, 6.0],
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
                                     parCWH=[0, 0.2],
                                     parBETAET=[0.3, 5],
                                     parC=[0, 1]
                                     )
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]

    def forward(self, x_hydro_model, c_hydro_model, params_raw, config, static_idx=-1,
                muwts=None, warm_up=0, init=False, routing=False, comprout=False,
                conv_params_hydro=None):
        nearzero = config['nearzero']
        nmul = config['nmul']

        # Initialization
        if config['hbvcap_no_warm']:
            warm_up = 0  # No warmup for HBV1.1p.
        
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                initmodel = HBVMulTDET().to(config['device'])
                Qsinit, SNOWPACK, MELTWATER, SM, SUZ, SLZ = initmodel(
                    xinit,
                    c_hydro_model,
                    params_raw,
                    config,
                    static_idx=warm_up-1,
                    muwts=None,
                    warm_up=0,
                    init=True,
                    routing=False,
                    comprout=False,
                    conv_params_hydro=None
                )
        else:
            # Without warm-up, initialize state variables with zeros.
            Ngrid = x_hydro_model.shape[1]
            SNOWPACK = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            MELTWATER = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            SM = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            SUZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])
            SLZ = (torch.zeros([Ngrid, nmul], dtype=torch.float32) + 0.001).to(config['device'])

        # Parameters
        params_dict_raw = dict()
        for num, param in enumerate(self.parameters_bound.keys()):
            params_dict_raw[param] = change_param_range(
                param=params_raw[:, :, num, :],
                bounds=self.parameters_bound[param]
            )
        
        # List of params to be made dynamic.
        if init:
            # Run all static for warmup.
            dy_params = []
        else:
            dy_params = config['dy_params']['HBV_capillary']

        vars = config['observations']['var_t_hydro_model']  # Forcing var names
        vars_c = config['observations']['var_c_hydro_model']  # Attribute var names

        # Forcings
        P = x_hydro_model[warm_up:, :, vars.index('prcp(mm/day)')]  # Precipitation
        T = x_hydro_model[warm_up:, :, vars.index('tmean(C)')]  # Mean air temp

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, nmul)

        # Get PET data.
        if config['pet_module'] == 'potet_hamon':
            # PET_coef = h.param_bounds_2D(PET_coef, 0, bounds=[0.004, 0.008], ndays=No_days, nmul=config['nmul'])
            # PET = get_potet(
            #     config=config, mean_air_temp=Tm, dayl=dayl, hamon_coef=PET_coef
            # )  # mm/day
            raise NotImplementedError

        elif config['pet_module'] == 'potet_hargreaves':
            day_of_year = x_hydro_model[warm_up:, :, vars.index('dayofyear')].unsqueeze(-1).repeat(1, 1, nmul)
            lat = c_hydro_model[:, vars_c.index('lat')].unsqueeze(0).unsqueeze(-1).repeat(day_of_year.shape[0], 1, nmul)
            Tmaxf = x_hydro_model[warm_up:, :, vars.index('tmax(C)')].unsqueeze(2).repeat(1, 1, nmul)
            Tminf = x_hydro_model[warm_up:, :, vars.index('tmin(C)')].unsqueeze(2).repeat(1, 1, nmul)

            # AET = PET_coef * PET 
            # PET_coef converts PET to Actual ET.
            PET = get_potet(config=config, tmin=Tminf, tmax=Tmaxf, tmean=Tm, lat=lat,
                            day_of_year=day_of_year)

        elif config['pet_module'] == 'dataset':
            # AET = PET_coef * PET
            # PET_coef converts PET to Actual ET
            PET = x_hydro_model[warm_up:, :, vars.index(config['pet_dataset_name'])]

        PETm = PET.unsqueeze(-1).repeat(1, 1, nmul)
        Nstep, Ngrid = P.size()

        # Apply correction factor to precipitation
        # P = parPCORR.repeat(Nstep, 1) * P

        # Initialize time series of model variables in shape [time, basins, nmul].
        Qsimmu = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.001).to(config['device'])
        Q0_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(config['device'])
        Q1_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(config['device'])
        Q2_sim = (torch.zeros(Pm.size(), dtype=torch.float32) + 0.0001).to(config['device'])

        AET = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])
        recharge_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])
        excs_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])
        evapfactor_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])
        tosoil_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])
        PERC_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])
        SWE_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])
        capillary_sim = (torch.zeros(Pm.size(), dtype=torch.float32)).to(config['device'])

        # Init static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in dy_params: # and len(params_raw.shape) > 2:
                params_dict[key] = params_dict_raw[key][static_idx, :, :]

        # Init dynamic parameters
        # (Use a dydrop ratio: fix a probability mask for setting dynamic params
        # as static in some basins.)
        if len(dy_params) > 0:
            params_dict_raw_dy = dict()
            pmat = torch.ones([Ngrid, 1]) * config['dy_drop']
            for i, key in enumerate(dy_params):
                drmask = torch.bernoulli(pmat).detach_().to(config['device'])
                dynPar = params_dict_raw[key]
                staPar = params_dict_raw[key][static_idx, :, :].unsqueeze(0).repeat([dynPar.shape[0], 1, 1])
                params_dict_raw_dy[key] = dynPar * (1 - drmask) + staPar * drmask

        for t in range(Nstep):
            # Get dynamic parameter values per timestep.
            for key in dy_params:
                params_dict[key] = params_dict_raw_dy[key][warm_up + t, :, :]

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
            SM = SM - excess
            # NOTE: Different from HBV 1.0. Add static/dynamicET shape parameter parBETAET.
            evapfactor = (SM / (params_dict['parLP'] * params_dict['parFC'])) ** params_dict['parBETAET']
            evapfactor = torch.clamp(evapfactor, min=0.0, max=1.0)
            ETact = PETm[t, :, :] * evapfactor
            ETact = torch.min(SM, ETact)
            SM = torch.clamp(SM - ETact, min=nearzero)  # SM != 0 for grad tracking.

            # Capillary rise (HBV 1.1p mod) -------------------------------
            capillary = torch.min(SLZ, params_dict['parC'] * SLZ * (1.0 - torch.clamp(SM / params_dict['parFC'], max=1.0)))

            SM = torch.clamp(SM + capillary, min=nearzero)
            SLZ = torch.clamp(SLZ - capillary, min=nearzero)

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
            capillary_sim[t, :, :] = capillary

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
        if routing:
            # Routing for all components or just the average.
            if comprout:
                # All components; reshape to [time, gages * num models]
                Qsim = Qsimmu.view(Nstep, Ngrid * nmul)
            else:
                # Average, then do routing.
                Qsim = Qsimavg

            # Scale routing params
            temp_a = change_param_range(
                param=conv_params_hydro[:, 0],
                bounds=self.conv_routing_hydro_model_bound[0]
            )
            temp_b = change_param_range(
                param=conv_params_hydro[:, 1],
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
                Qstemp = Qsrout.view(Nstep, Ngrid, nmul)
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

        if init:
            # Only return model states for warmup.
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
                        capillary=capillary_sim.mean(-1, keepdim=True)
                        )
