import torch

from hydroDL2.core.calc import change_param_range
from hydroDL2.core.calc.uh_routing import UH_conv, UH_gamma


class PRMS(torch.nn.Module):
    """Multi-component Pytorch PRMS model.

    Adapted from Farshid Rahmani.
    """
    def __init__(self):
        super(PRMS, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.parameters_bound = dict(
            tt=[-3, 5],    # tt, Temperature threshold for snowfall and melt [oC]
            ddf=[0, 20],    # ddf,  Degree-day factor for snowmelt [mm/oC/d]
            alpha=[0, 1],     # alpha, Fraction of rainfall on soil moisture going to interception [-]
            beta=[0, 1],    # beta, Fraction of catchment where rain goes to soil moisture [-]
            stor=[0, 5],    # stor, Maximum interception capcity [mm]
            retip=[0, 50],    # retip, Maximum impervious area storage [mm]
            fscn=[0, 1],    # fscn, Fraction of SCX where SCN is located [-]
            scx=[0, 1],    # scx, Maximum contributing fraction area to saturation excess flow [-]
            flz=[0.005, 0.995],    # flz, Fraction of total soil moisture that is the lower zone [-]
            stot=[1, 2000],    # stot, Total soil moisture storage [mm]: REMX+SMAX
            cgw=[0, 20],    # cgw, Constant drainage to deep groundwater [mm/d]
            resmax=[1, 300],    # resmax, Maximum flow routing reservoir storage (used for scaling only, there is no overflow) [mm]
            k1=[0, 1],    # k1, Groundwater drainage coefficient [d-1]
            k2=[1, 5],    # k2, Groundwater drainage non-linearity [-]
            k3=[0, 1],    # k3, Interflow coefficient 1 [d-1]
            k4=[0, 1],    # k4, Interflow coefficient 2 [mm-1 d-1]
            k5=[0, 1],    # k5, Baseflow coefficient [d-1]
            k6=[0, 1],    # k6, Groundwater sink coefficient [d-1],
        )
        self.conv_routing_hydro_model_bound = [
            [0, 2.9],  # routing parameter a
            [0, 6.5]   # routing parameter b
        ]
        # PET_coef for converting PET to AET (Farshid).
        self.PET_coef_bound = [
            [0.01, 1]
        ]

    def forward(self, x_hydro_model, params_raw, config, static_idx=-1,
                warm_up=0, init=False, routing=False, conv_params_hydro=None):
        nearzero = config['nearzero']
        nmul = config['nmul']

        vars = config['observations']["var_t_hydro_model"]

        # Initialization
        if warm_up > 0:
            with torch.no_grad():
                xinit = x_hydro_model[0:warm_up, :, :]
                # paramsinit = params[:, :warm_up, :]
                # PET_coefinit = PET_coef[:, :warm_up, :]
                initmodel = PRMS().to(config["device"])
                Q_init, snow_storage, XIN_storage, RSTOR_storage, \
                    RECHR_storage, SMAV_storage, \
                    RES_storage, GW_storage = initmodel(
                        xinit,
                        params_raw,
                        config,
                        static_idx=warm_up-1,
                        warm_up=0,
                        init=True,
                        routing=False,
                        conv_params_hydro=None
                    )
        else:
            # Without warm-up, initialize state variables with zeros.
            Ngrid = x_hydro_model.shape[1]

            # snow storage
            snow_storage = torch.zeros([Ngrid, nmul], dtype=torch.float32,
                                       device=config["device"]) + 0.001
            # interception storage
            XIN_storage = torch.zeros([Ngrid, nmul], dtype=torch.float32,
                                      device=config["device"]) + 0.001
            # RSTOR storage
            RSTOR_storage = torch.zeros([Ngrid, nmul], dtype=torch.float32,
                                        device=config["device"]) + 0.001
            # storage in upper soil moisture zone
            RECHR_storage = torch.zeros([Ngrid, nmul], dtype=torch.float32,
                                        device=config["device"]) + 0.001
            # storage in lower soil moisture zone
            SMAV_storage = torch.zeros([Ngrid, nmul], dtype=torch.float32,
                                       device=config["device"]) + 0.001
            # storage in runoff reservoir
            RES_storage = torch.zeros([Ngrid, nmul], dtype=torch.float32,
                                      device=config["device"]) + 0.001
            # GW storage
            GW_storage = torch.zeros([Ngrid, nmul], dtype=torch.float32,
                                     device=config["device"]) + 0.001

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
            dy_params = config['dy_params']['marrmot_PRMS']

        P = x_hydro_model[warm_up:, :, vars.index('prcp')]  # Precipitation
        T = x_hydro_model[warm_up:, :, vars.index('tmean')]  # Mean air temp
        PET = x_hydro_model[warm_up:, :, vars.index('pet')] # Potential ET

        # Expand dims to accomodate for nmul models.
        Pm = P.unsqueeze(2).repeat(1, 1, nmul)
        Tm = T.unsqueeze(2).repeat(1, 1, nmul)
        PETm = PET.unsqueeze(-1).repeat(1, 1, nmul)

        Nstep, Ngrid = P.size()

        # AET = PET_coef * PET
        # initialize the Q_sim and other fluxes
        Q_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        sas_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        sro_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        bas_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        ras_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        snk_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        AET = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        inf_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        PC_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        SEP_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        GAD_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        ea_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        qres_sim = torch.zeros(Pm.shape, dtype=torch.float32, device=config["device"])
        
        # Init static parameters
        params_dict = dict()
        for key in params_dict_raw.keys():
            if key not in dy_params: # and len(params_raw.shape) > 2:
                # Use the last day of data as static parameter's value.
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

            scn = params_dict["fscn"] * params_dict["scx"]
            remx = (1 - params_dict["flz"]) * params_dict["stot"]
            smax = params_dict["flz"] * params_dict["stot"]

            delta_t = 1  # timestep (day)
            PRECIP = Pm[t, :, :]
            Ep = PETm[t, :, :]
            temp = Tm[t, :, :]

            # Fluxes
            flux_ps = torch.mul(PRECIP, (temp <= params_dict['tt']).type(torch.float32))
            flux_pr = torch.mul(PRECIP, (temp > params_dict['tt']).type(torch.float32))
            snow_storage = snow_storage + flux_ps
            flux_m = torch.clamp(params_dict["ddf"] * (temp - params_dict["tt"]), min=0.0)
            flux_m = torch.min(flux_m, snow_storage/delta_t)
            # flux_m = torch.clamp(flux_m, min=0.0)
            snow_storage = torch.clamp(snow_storage - flux_m, min=nearzero)

            flux_pim = flux_pr * (1 - params_dict["beta"])
            flux_psm = flux_pr * params_dict["beta"]
            flux_pby = flux_psm * (1 - params_dict["alpha"])
            flux_pin = flux_psm * params_dict["alpha"]

            XIN_storage = XIN_storage + flux_pin
            flux_ptf = XIN_storage - params_dict["stor"]
            flux_ptf = torch.clamp(flux_ptf, min=0.0)
            XIN_storage = torch.clamp(XIN_storage - flux_ptf, min=nearzero)
            evap_max_in = Ep * params_dict["beta"]   # only can happen in pervious area
            flux_ein = torch.min(evap_max_in, XIN_storage/delta_t)
            XIN_storage = torch.clamp(XIN_storage - flux_ein, min=nearzero)

            flux_mim = flux_m * (1 - params_dict["beta"])
            flux_msm = flux_m * params_dict["beta"]
            RSTOR_storage = RSTOR_storage + flux_mim + flux_pim
            flux_sas = RSTOR_storage - params_dict["retip"]
            flux_sas = torch.clamp(flux_sas, min=0.0)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_sas, min=nearzero)
            evap_max_im = (1 - params_dict["beta"]) * Ep
            flux_eim = torch.min(evap_max_im, RSTOR_storage / delta_t)
            RSTOR_storage = torch.clamp(RSTOR_storage - flux_eim, min=nearzero)

            sro_lin_ratio = scn + (params_dict["scx"] - scn) * (RECHR_storage / remx)
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
            RECHR_storage = torch.clamp(RECHR_storage - flux_ea, min=nearzero)

            SMAV_storage = SMAV_storage + flux_pc
            flux_excs = SMAV_storage - smax
            flux_excs = torch.clamp(flux_excs, min=0.0)
            SMAV_storage = SMAV_storage - flux_excs
            transp = torch.where(RECHR_storage < (Ep - flux_ein - flux_eim),
                                 (SMAV_storage/smax) * (Ep - flux_ein - flux_eim - flux_ea),
                                 torch.zeros(flux_excs.shape, dtype=torch.float32, device=config["device"]))
            transp = torch.clamp(transp, min=0.0)    # in case Ep - flux_ein - flux_eim - flux_ea was negative
            SMAV_storage = torch.clamp(SMAV_storage - transp, min=nearzero)

            flux_sep = torch.min(params_dict["cgw"], flux_excs)
            flux_qres = torch.clamp(flux_excs - flux_sep, min=0.0)

            RES_storage = RES_storage + flux_qres
            flux_gad = params_dict["k1"] * ((RES_storage / params_dict['resmax']) ** params_dict["k2"])
            flux_gad = torch.min(flux_gad, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_gad, min=nearzero)
            flux_ras = params_dict["k3"] * RES_storage + params_dict["k4"] * (RES_storage ** 2)
            flux_ras = torch.min(flux_ras, RES_storage)
            RES_storage = torch.clamp(RES_storage - flux_ras, min=nearzero)
            # RES_excess = RES_storage - resmax[:, t, :]   # if there is still overflow, it happend in discrete version
            # RES_excess = torch.clamp(RES_excess, min=0.0)
            # flux_ras = flux_ras + RES_excess
            # RES_storage = torch.clamp(RES_storage - RES_excess, min=nearzero)


            GW_storage = GW_storage + flux_gad + flux_sep
            flux_bas = params_dict["k5"] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_bas, min=nearzero)
            flux_snk = params_dict["k6"] * GW_storage
            GW_storage = torch.clamp(GW_storage - flux_snk, min=nearzero)

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

        if routing == True:
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


        if init:
            # Only return model states for warmup.
            return Qsrout, snow_storage, XIN_storage, RSTOR_storage, \
                RECHR_storage, SMAV_storage, RES_storage, GW_storage
        else:
            return dict(flow_sim=Qsrout,
                        srflow=Qsas_rout + Qsro_rout,
                        ssflow=Qras_rout,
                        gwflow=Qbas_rout,
                        sink=torch.mean(snk_sim, -1).unsqueeze(-1),
                        PET_hydro=PET.mean(-1, keepdim=True),
                        AET_hydro=AET.mean(-1, keepdim=True),
                        flow_sim_no_rout=Q_sim.mean(-1, keepdim=True),
                        srflow_no_rout=(sas_sim + sro_sim).mean(-1, keepdim=True),
                        ssflow_no_rout=ras_sim.mean(-1, keepdim=True),
                        gwflow_no_rout=bas_sim.mean(-1, keepdim=True),
                        flux_inf=inf_sim.mean(-1, keepdim=True),
                        flux_pc=PC_sim.mean(-1, keepdim=True),
                        flux_sep=SEP_sim.mean(-1, keepdim=True),
                        flux_gad=GAD_sim.mean(-1, keepdim=True),
                        flux_ea=ea_sim.mean(-1, keepdim=True),
                        flux_qres=qres_sim.mean(-1, keepdim=True),
                        )
