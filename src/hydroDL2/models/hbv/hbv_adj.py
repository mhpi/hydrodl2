from typing import Any, Optional, Union

import sourcedefender
import torch

from hydroDL2.core.calc import change_param_range
from hydroDL2.core.calc.batch_jacobian import batchJacobian
from hydroDL2.core.calc.FDJacobian import (finite_difference_jacobian,
                                           finite_difference_jacobian_P)
from hydroDL2.core.calc.uh_routing import UH_conv, UH_gamma


class HBVAdjoint(torch.nn.Module):
    """
    Multi-component PyTorch HBV model using implicit numerical scheme and
    gradient tracking supported by adjoint method.
    
    Author
    ------
    Yalan Song

    Publication
    -----------
    -   Song, Y., Knoben, W. J. M., Clark, M. P., Feng, D., Lawson, K. E., & Shen,
        C. (2024). When ancient numerical demons meet physics-informed machine
        learning: Adjoint-based gradients for implicit differentiable modeling.
        Hydrology and Earth System Sciences Discussions, 1-35.
        https://doi.org/10.5194/hess-2023-258

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
            device: Optional[torch.device] = None
        ) -> None:
        super().__init__()
        self.name = 'HBV Adjoint'
        self.config = config
        self.initialize = False
        self.warm_up = 0
        self.dynamic_params = []
        self.dy_drop = 0.0
        self.variables = ['prcp', 'tmean', 'pet']
        self.routing = True
        self.comprout = False
        self.nearzero = 1e-5
        self.nmul = 1
        self.ad_efficient = True
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
            self.dy_drop = config.get('dy_drop', self.dy_drop)
            self.dynamic_params = config['dynamic_params'].get('HBV_adj', self.dynamic_params)
            self.variables = config.get('variables', self.variables)
            self.routing = config.get('routing', self.routing)
            self.comprout = config.get('comprout', self.comprout)
            self.nearzero = config.get('nearzero', self.nearzero)
            self.nmul = config.get('nmul', self.nmul)
            self.ad_efficient = config.get('ad_efficient', self.ad_efficient)
            if 'parBETAET' in self.dynamic_params :
                self.parameter_bounds['parBETAET'] = [0.3, 5]

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
            n_steps: int,
            n_grid: int,
        ) -> dict[str, torch.Tensor]:
        """Extract physical model and routing parameters from NN output.
        
        Parameters
        ----------
        parameters
            Unprocessed, learned parameters from a neural network.
        n_steps
            Number of time steps in the input data.
        n_grid
            Number of grid cells in the input data.
        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
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
        ## Merge the multi-components into batch dimension for parallel Jacobian
        phy_params = phy_params.permute([0,3,1,2])  
        bsnew = n_grid * self.nmul
        phy_params = phy_params.reshape(n_steps, bsnew, phy_param_count)        

        # Routing parameters
        if self.routing == True:
            routing_params = torch.sigmoid(
                parameters[-1, :, phy_param_count * self.nmul:]
            )
        return phy_params, routing_params

    def make_phy_parameters(
            self,
            phy_params: torch.Tensor,
            name_list: list,
            dy_list:list,
        ) -> torch.Tensor:
        """Descale physical parameters.
        
        Parameters
        ----------
        phy_params
            Normalized physical parameters.
        name_list
            List of physical parameter names.
        dy_list
            List of dynamic parameters.
        
        Returns
        -------
        torch.Tensor
            Tensor of physical parameters.
        """
        n_steps, n_grid, nfea = phy_params.size()
        parstaFull = phy_params[-1, :, :].unsqueeze(0).repeat([n_steps, 1, 1])
        # Precompute probability mask for dynamic parameters
        if dy_list:
            pmat = torch.ones([1,n_grid]) * self.dy_drop
            parhbvFull = torch.clone(parstaFull)
            for i, name in enumerate(name_list):
                if name in dy_list:
                    staPar = parstaFull[:, :, i]
                    dynPar = phy_params[:, :, i]
                    drmask = torch.bernoulli(pmat).detach_().cuda()  # to drop some dynamic parameters as static
                    comPar = dynPar * (1 - drmask) + staPar * drmask
                    parhbvFull[:, :, i] = comPar
            return parhbvFull

        else:
            return parstaFull

    def descale_rout_parameters(
            self,
            rout_params: torch.Tensor,
            name_list: list,
        ) -> torch.Tensor:
        """Descale routing parameters.
        
        Parameters
        ----------
        rout_params
            Normalized routing parameters.
        name_list
            List of routing parameter names.

        Returns
        -------
        dict
            Dictionary of descaled routing parameters.
        """

        parameter_dict = {}
        for i, name in enumerate(name_list):
            param = rout_params[:, i]

            parameter_dict[name] = change_param_range(
                param=param,
                bounds=self.routing_parameter_bounds[name]
            )
        return parameter_dict

    def forward(
            self,
            x_dict: dict[str, torch.Tensor],
            parameters: torch.Tensor
        ) -> Union[tuple, dict[str, torch.Tensor]]:
        """Forward pass for HBV Adj.
        
        Parameters
        ----------
        x_dict
            Dictionary of input forcing data.
        parameters
            Unprocessed, learned parameters from a neural network.
        
        Returns
        -------
        Union[tuple, dict]
            Tuple or dictionary of model outputs.
        """
        # Unpack input data.
        x = x_dict['x_phy']

        n_steps,bs,_ = x.size()
        bsnew = bs * self.nmul
        phy_params, routing_params = self.unpack_parameters(parameters, n_steps, bs)

        nS = 5 ## For this version of HBV, we have 5 state varibales
        y_init = torch.zeros((bsnew, nS)).to(self.device)
        nflux = 1 # currently only return streamflow
        delta_t  = torch.tensor(1.0).to(device = self.device)  ## Daily model
        if self.warm_up> 0:
            phy_params_warmup = self.make_phy_parameters(phy_params[:self.warm_up,:,:],self.phy_param_names,[])
            x_warmup = x[:self.warm_up,:,:].unsqueeze(1).repeat([1, self.nmul, 1, 1])
            x_warmup = x_warmup.view(x_warmup.shape[0], bsnew, x_warmup.shape[-1])            
            f_warm_up = HBV(x_warmup, self.parameter_bounds)
            M_warm_up = MOL(f_warm_up, nS, nflux, self.warm_up, bsDefault=bsnew, mtd=0, dtDefault=delta_t,ad_efficient=self.ad_efficient)
            y0 = M_warm_up.nsteps_pDyn(phy_params_warmup, y_init)[-1, :, :]
        else:
            y0 = y_init
        
        phy_params_run = self.make_phy_parameters(phy_params[self.warm_up:,:,:],self.phy_param_names,self.dynamic_params)
        routy_params_dict = self.descale_rout_parameters(routing_params,self.rout_params_name)
        
        xTrain = x[self.warm_up:,:,:].unsqueeze(1).repeat([1, self.nmul, 1, 1])
        xTrain = xTrain.view(xTrain.shape[0], bsnew, xTrain.shape[-1]) 
        # Without warm-up, initialize state variables with zeros.
        
        nt = phy_params_run.shape[0]
        
        simulation = torch.zeros((nt, bsnew, nflux)).to(self.device)

        f = HBV(xTrain, self.parameter_bounds)
        
        M = MOL(f, nS, nflux, nt, bsDefault=bsnew, dtDefault=delta_t, mtd=0,ad_efficient=self.ad_efficient)
  
        ### Newton iterations with adjoint
        ySolution = M.nsteps_pDyn(phy_params_run, y0)

        for day in range(0, nt):
            _, flux = f(ySolution[day, :, :], phy_params_run[day, :, :], day, expanded_num = [1])
            simulation[day, :, :] = flux * delta_t


        if self.nmul > 1 :
            simulation = simulation.view(nt,self.nmul,bs,nflux)
            simulation = simulation.mean(dim=1)

        routa = routy_params_dict['rout_a'].unsqueeze(0).repeat(nt, 1).unsqueeze(-1) 
        routb = routy_params_dict['rout_b'].unsqueeze(0).repeat(nt, 1).unsqueeze(-1)

        UH = UH_gamma(routa, routb, lenF=15)  # lenF: folter
        rf = simulation.permute([1, 2, 0])  # dim:gage*var*time
        UH = UH.permute([1, 2, 0])  # dim: gage*var*time
        Qsrout = UH_conv(rf, UH).permute([2, 0, 1])

        # Return all sim results.
        return {
            'flow_sim': Qsrout 
        }

class HBV(torch.nn.Module):
    def __init__(self, climate_data,parameter_bounds):
        super().__init__()
        self.climate_data = climate_data
        self.parameter_bounds = parameter_bounds

    def forward(self, y,theta,t,expanded_num,returnFlux=False,aux=None):
        ##parameters

        Beta = self.parameter_bounds['parBETA'][0] + theta[:,0]*(self.parameter_bounds['parBETA'][1]-self.parameter_bounds['parBETA'][0])
        FC = self.parameter_bounds['parFC'][0] + theta[:,1]*(self.parameter_bounds['parFC'][1]-self.parameter_bounds['parFC'][0])
        K0 = self.parameter_bounds['parK0'][0] + theta[:,2]*(self.parameter_bounds['parK0'][1]-self.parameter_bounds['parK0'][0])
        K1 = self.parameter_bounds['parK1'][0] + theta[:,3]*(self.parameter_bounds['parK1'][1]-self.parameter_bounds['parK1'][0])
        K2 = self.parameter_bounds['parK2'][0] + theta[:,4]*(self.parameter_bounds['parK2'][1]-self.parameter_bounds['parK2'][0])
        LP = self.parameter_bounds['parLP'][0] + theta[:,5]*(self.parameter_bounds['parLP'][1]-self.parameter_bounds['parLP'][0])
        PERC = self.parameter_bounds['parPERC'][0] + theta[:,6]*(self.parameter_bounds['parPERC'][1]-self.parameter_bounds['parPERC'][0])
        UZL = self.parameter_bounds['parUZL'][0] + theta[:,7]*(self.parameter_bounds['parUZL'][1]-self.parameter_bounds['parUZL'][0])
        TT = self.parameter_bounds['parTT'][0] + theta[:,8]*(self.parameter_bounds['parTT'][1]-self.parameter_bounds['parTT'][0])
        CFMAX = self.parameter_bounds['parCFMAX'][0] + theta[:,9]*(self.parameter_bounds['parCFMAX'][1]-self.parameter_bounds['parCFMAX'][0])
        CFR = self.parameter_bounds['parCFR'][0] + theta[:,10]*(self.parameter_bounds['parCFR'][1]-self.parameter_bounds['parCFR'][0])
        CWH = self.parameter_bounds['parCWH'][0] + theta[:,11]*(self.parameter_bounds['parCWH'][1]-self.parameter_bounds['parCWH'][0])
        BETAET = self.parameter_bounds['parBETAET'][0] + theta[:,12]*(self.parameter_bounds['parBETAET'][1]-self.parameter_bounds['parBETAET'][0])

        PRECS = 0
        ##% stores
        SNOWPACK = torch.clamp(y[:,0] , min=PRECS)  #SNOWPACK
        MELTWATER = torch.clamp(y[:,1], min=PRECS)  #MELTWATER
        SM = torch.clamp(y[:,2], min=1e-8)   #SM
        SUZ = torch.clamp(y[:,3] , min=PRECS) #SUZ
        SLZ = torch.clamp(y[:,4], min=PRECS)   #SLZ
        dS = torch.zeros(y.shape[0],y.shape[1]).to(y)
        fluxes = torch.zeros((y.shape[0],1)).to(y)

        climate_in0 = self.climate_data[int(t),:,:];   ##% climate at this step

        for idx, expanded_num_i in enumerate(expanded_num):
            if idx == 0:
                climate_in =climate_in0.repeat_interleave(expanded_num_i, dim=0)
            else:
                climate_in = torch.cat([climate_in, climate_in0.repeat_interleave(expanded_num_i, dim=0)],dim = 0)

        P  = climate_in[:,0]
        Ep = climate_in[:,2]
        T  = climate_in[:,1]

        ##% fluxes functions
        flux_sf   = self.snowfall(P,T,TT)
        flux_refr = self.refreeze(CFR,CFMAX,T,TT,MELTWATER)
        flux_melt = self.melt(CFMAX,T,TT,SNOWPACK)
        flux_rf   = self.rainfall(P,T,TT)
        flux_Isnow   =  self.Isnow(MELTWATER,CWH,SNOWPACK)
        flux_PEFF   = self.Peff(SM,FC,Beta,flux_rf,flux_Isnow)
        flux_ex   = self.excess(SM,FC)
        flux_et   = self.evap(SM,FC,LP,Ep,BETAET)
        flux_perc = self.percolation(PERC,SUZ)
        flux_q0   = self.interflow(K0,SUZ,UZL)
        flux_q1   = self.baseflow(K1,SUZ)
        flux_q2   = self.baseflow(K2,SLZ)

        #% stores ODEs
        dS[:,0] = flux_sf + flux_refr - flux_melt
        dS[:,1] = flux_melt - flux_refr - flux_Isnow
        dS[:,2] = flux_Isnow + flux_rf - flux_PEFF - flux_ex - flux_et 
        dS[:,3] = flux_PEFF + flux_ex - flux_perc - flux_q0 - flux_q1
        dS[:,4] = flux_perc - flux_q2 

        fluxes[:,0] =flux_q0 + flux_q1 + flux_q2

        if returnFlux:
            return fluxes,flux_q0.unsqueeze(-1),flux_q1.unsqueeze(-1),flux_q2.unsqueeze(-1),flux_et.unsqueeze(-1)
        else:
            return dS,fluxes
 

    def snowfall(self,P,T,TT):
        return torch.mul(P, (T < TT))

    def refreeze(self,CFR,CFMAX,T,TT,MELTWATER):
        refreezing = CFR * CFMAX * (TT - T)
        refreezing = torch.clamp(refreezing, min=0.0)
        return torch.min(refreezing, MELTWATER)

    def melt(self,CFMAX,T,TT,SNOWPACK):
        melt = CFMAX * (T - TT)
        melt = torch.clamp(melt, min=0.0)
        return torch.min(melt, SNOWPACK)

    def rainfall(self,P,T,TT):

        return torch.mul(P, (T >= TT))
    def Isnow(self,MELTWATER,CWH,SNOWPACK):
        tosoil = MELTWATER - (CWH * SNOWPACK)
        tosoil = torch.clamp(tosoil, min=0.0)
        return tosoil

    def Peff(self,SM,FC,Beta,flux_rf,flux_Isnow):
        soil_wetness = (SM / FC) ** Beta
        soil_wetness = torch.clamp(soil_wetness, min=0.0, max=1.0)
        return (flux_rf + flux_Isnow) * soil_wetness

    def excess(self,SM,FC):
        excess = SM - FC
        return  torch.clamp(excess, min=0.0)

    def evap(self,SM,FC,LP,Ep,BETAET):
        evapfactor = (SM / (LP * FC)) ** BETAET
        evapfactor  = torch.clamp(evapfactor, min=0.0, max=1.0)
        ETact = Ep * evapfactor
        return torch.min(SM, ETact)

    def interflow(self,K0,SUZ,UZL):
        return K0 * torch.clamp(SUZ - UZL, min=0.0)
    def percolation(self,PERC,SUZ):
        return torch.min(SUZ, PERC)

    def baseflow(self,K,S):
        return K * S


matrixSolve = torch.linalg.solve


class NewtonSolve(torch.autograd.Function):
  @staticmethod
  def forward(ctx, p, p2, t,G, x0=None, auxG=None, batchP=True,eval = False,ad_efficient = True):
    useAD_jac=True
    if x0 is None and p2 is not None:
      x0 = p2

    x = x0.clone().detach(); i=0
    max_iter=3; gtol=1e-3

    if useAD_jac:
        torch.set_grad_enabled(True)

    x.requires_grad = True

    if p2 is None:
        gg = G(x, p, t, auxG)
    else:
        gg = G(x, p, p2, t, auxG)
    if ad_efficient:
        dGdx = batchJacobian(gg,x,graphed=True)
    else:
        dGdx = batchJacobian_AD_slow(gg, x, graphed=True)
    if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
        raise RuntimeError(f"Jacobian matrix is NaN")
    x = x.detach()

    torch.set_grad_enabled(False)
    resnorm = torch.linalg.norm(gg, float('inf'),dim= [1]) # calculate norm of the residuals
    resnorm0 = 100*resnorm;


    while ((torch.max(resnorm)>gtol ) and  i<=max_iter):
        i+=1
        if torch.max(resnorm/resnorm0) > 0.2:
              if useAD_jac:
                torch.set_grad_enabled(True)

              x.requires_grad = True

              if p2 is None:
                gg = G(x, p, t, auxG)
              else:
                gg = G(x, p, p2, t, auxG)
              if ad_efficient:
                dGdx = batchJacobian(gg,x,graphed=True)
              else:
                dGdx = batchJacobian_AD_slow(gg, x, graphed=True)
              if torch.isnan(dGdx).any() or torch.isinf(dGdx).any():
                raise RuntimeError(f"Jacobian matrix is NaN")

              x = x.detach()

              torch.set_grad_enabled(False)

        if dGdx.ndim==gg.ndim: # same dimension, must be scalar.
          dx =  (gg/dGdx).detach()
        else:
          dx =  matrixSolve(dGdx, gg).detach()
        x = x - dx
        if useAD_jac:
            torch.set_grad_enabled(True)
        x.requires_grad = True
        if p2 is None:
            gg = G(x, p, t, auxG)
        else:
            gg = G(x, p, p2, t, auxG)
        torch.set_grad_enabled(False)
        resnorm0 = resnorm; ##% old resnorm
        resnorm = torch.linalg.norm(gg, float('inf'),dim= [1]);

    torch.set_grad_enabled(True)
    x = x.detach()
    if not eval:
        if batchP:
          # dGdp is needed only upon convergence.
          if p2 is None:
            if ad_efficient:
                dGdp = batchJacobian(gg, p, graphed=True); dGdp2 = None
            else:
                dGdp = batchJacobian_AD_slow(gg, p, graphed=True);
                dGdp2 = None
          else:
            if ad_efficient:
                dGdp, dGdp2 = batchJacobian(gg, (p,p2),graphed=True)
            else:
                dx = matrixSolve(dGdx, gg)
            x = x - dx
            gg = G(x, p, p2, t, [1], auxG)
            resnorm0 = resnorm;  ##% old resnorm
            resnorm = torch.linalg.norm(gg, float('inf'), dim=[1]);
    
        if batchP:
            dGdp,dGdp2 =  finite_difference_jacobian_P( G, x, p, p2, t, epsilon, auxG)
        
        else:
            assert ("nonbatchp (like NN) pathway not debugged through yet")
         # print("day ", t, "Iterations ", i)
        ctx.save_for_backward(dGdp.float(), dGdp2.float(), dGdx.float())
         # This way, we reduced one forward run. You can also save these two to the CPU if forward run is
         # Alternatively, if memory is a problem, save x and run g during the backward.
        del gg
        return x.float()

    @staticmethod
    def backward(ctx, dLdx):
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)
        with torch.no_grad():
            dGdp, dGdp2, dGdx = ctx.saved_tensors
            dGdxT = torch.permute(dGdx, (0, 2, 1))
            lambTneg = matrixSolve(dGdxT, dLdx);
            if lambTneg.ndim <= 2:
                lambTneg = torch.unsqueeze(lambTneg, 2)
            dLdp = -torch.bmm(torch.permute(lambTneg, (0, 2, 1)), dGdp)
            dLdp = torch.squeeze(dLdp, 1)  # ADHOC!! DON"T KNOW WHY!!
            if dGdp2 is None:
                dLdp2 = None
            else:
                dLdp2 = -torch.bmm(torch.permute(lambTneg, (0, 2, 1)), dGdp2)
                dLdp2 = torch.squeeze(dLdp2, 1)  # ADHOC!! DON"T KNOW WHY!!
            return dLdp, dLdp2, None, None, None, None, None, None


class MOL(torch.nn.Module):
  # Method of Lines time integrator as a nonlinear equation G(x, p, xt, t, auxG)=0.
  # rhs is preloaded at construct and is the equation for the right hand side of the equation.
    def __init__(self, rhsFunc,ny,nflux,rho, bsDefault =1 , mtd = 0, dtDefault=0, solveAdj = NewtonSolve.apply,eval = False,ad_efficient=True):
        super(MOL, self).__init__()
        self.mtd = mtd # time discretization method. =0 for backward Euler
        self.rhs = rhsFunc
        self.delta_t = dtDefault
        self.bs = bsDefault
        self.ny = ny
        self.nflux = nflux
        self.rho = rho
        self.solveAdj = solveAdj
        self.eval = eval
        self.ad_efficient = ad_efficient

    def forward(self, x, p, xt, t, expand_num,auxG ): # take one step
        # xt is x^{t}. trying to solve for x^{t+1}
        dt, aux = auxG # expand auxiliary data

        if self.mtd == 0: # backward Euler
          rhs,_ = self.rhs(x, p, t, expand_num, returnFlux=False,aux = aux) # should return [nb,ng]
          gg = (x - xt)/dt - rhs
        elif self.mtd == 1: # Crank Nicholson
          rhs,_  = self.rhs(x, p, t,expand_num,returnFlux=False, aux = aux) # should return [nb,ng]
          rhst,_ = self.rhs(xt, p, t,expand_num,returnFlux=False, aux = aux) # should return [nb,ng]
          gg = (x - xt)/dt - (rhs+rhst)*0.5
        return gg

    def nsteps_pDyn(self,pDyn,x0):
        bs = self.bs
        ny = self.ny
        delta_t = self.delta_t
        rho = self.rho
        ySolution = torch.zeros((rho,bs,ny)).to(pDyn)
        ySolution[0,:,:] = x0

        xt=x0.clone().requires_grad_()

        auxG = (delta_t, None)

        for t in range(rho):
            p = pDyn[t,:,:]

            x = self.solveAdj(p, xt,t, self.forward, None, auxG,True, self.eval,self.ad_efficient)

            ySolution[t,:,:]  = x
            xt = x

        return ySolution
