import torch
import torch.nn.functional as F



def UH_gamma(a, b, lenF=10):
    """ TODO: Revise"""

    # UH. a [time (same all time steps), batch, var]
    m = a.shape
    lenF = min(a.shape[0], lenF)
    w = torch.zeros([lenF, m[1], m[2]])
    aa = F.relu(a[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.1  # minimum 0.1. First dimension of a is repeat
    theta = F.relu(b[0:lenF, :, :]).view([lenF, m[1], m[2]]) + 0.5  # minimum 0.5
    t = torch.arange(0.5, lenF * 1.0).view([lenF, 1, 1]).repeat([1, m[1], m[2]])
    t = t.cuda(aa.device)
    denom = (aa.lgamma().exp()) * (theta ** aa)
    mid = t ** (aa - 1)
    right = torch.exp(-t / theta)
    w = 1 / denom * mid * right
    w = w / w.sum(0)  # scale to 1 for each UH

    return w

def UH_conv(x, UH, viewmode=1):
    """
    TODO: Revise
    
    UH is a vector indicating the unit hydrograph
    the convolved dimension will be the last dimension
    UH convolution is
    Q(t)=\integral(x(\tao)*UH(t-\tao))d\tao
    conv1d does \integral(w(\tao)*x(t+\tao))d\tao
    hence we flip the UH
    https://programmer.group/pytorch-learning-conv1d-conv2d-and-conv3d.html
    view

    x: [batch, var, time]
    UH:[batch, var, uhLen]
    batch needs to be accommodated by channels and we make use of groups
    https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
    https://pytorch.org/docs/stable/nn.functional.html
    """
    mm = x.shape;
    nb = mm[0]
    m = UH.shape[-1]
    padd = m - 1
    if viewmode == 1:
        xx = x.view([1, nb, mm[-1]])
        w = UH.view([nb, 1, m])
        groups = nb

    y = F.conv1d(xx, torch.flip(w, [2]), groups=groups, padding=padd, stride=1, bias=None)
    if padd != 0:
        y = y[:, :, 0:-padd]
    return y.view(mm)


def source_flow_calculation(config, flow_out, c_NN, after_routing=True):
    """ TODO: Revise"""

    varC_NN = config['var_c_nn']
    if 'DRAIN_SQKM' in varC_NN:
        area_name = 'DRAIN_SQKM'
    elif 'area_gages2' in varC_NN:
        area_name = 'area_gages2'
    else:
        print("area of basins are not available among attributes dataset")
    area = c_NN[:, varC_NN.index(area_name)].unsqueeze(0).unsqueeze(-1).repeat(
        flow_out['flow_sim'].shape[
            0], 1, 1)
    # flow calculation. converting mm/day to m3/sec
    if after_routing == True:
        srflow = (1000 / 86400) * area * (flow_out['srflow']).repeat(1, 1, config['nmul'])  # Q_t - gw - ss
        ssflow = (1000 / 86400) * area * (flow_out['ssflow']).repeat(1, 1, config['nmul'])  # ras
        gwflow = (1000 / 86400) * area * (flow_out['gwflow']).repeat(1, 1, config['nmul'])
    else:
        srflow = (1000 / 86400) * area * (flow_out['srflow_no_rout']).repeat(1, 1, config['nmul'])  # Q_t - gw - ss
        ssflow = (1000 / 86400) * area * (flow_out['ssflow_no_rout']).repeat(1, 1, config['nmul'])  # ras
        gwflow = (1000 / 86400) * area * (flow_out['gwflow_no_rout']).repeat(1, 1, config['nmul'])
    # srflow = torch.clamp(srflow, min=0.0)  # to remove the small negative values
    # ssflow = torch.clamp(ssflow, min=0.0)
    # gwflow = torch.clamp(gwflow, min=0.0)
    return srflow, ssflow, gwflow
