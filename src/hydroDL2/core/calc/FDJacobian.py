import torch
def finite_difference_jacobian(G, x, p, p2, t, epsilon, auxG,perturbed_p = 0):


    nb, nx = x.shape



    if perturbed_p == 0:
        ny = nx
    elif perturbed_p == 1:
        ny = p.shape[1]
    elif perturbed_p == 2:
        ny = p2.shape[1]
    else:
        print("perturbed_p is not set corretly")
    xE = x.repeat_interleave(ny + 1, dim=0).double()
    pE = p.repeat_interleave(ny + 1, dim=0).double()
    p2E = p2.repeat_interleave(ny + 1, dim=0).double()

    perturbation = torch.eye(ny).unsqueeze(0).expand(nb, -1, -1) * epsilon
    perturbation = torch.cat([torch.zeros(nb, 1, ny), perturbation], dim=1).to(x)

    expand_num = ny + 1
    # Expand x and add perturbations
    if perturbed_p == 0:
        xE = xE + perturbation.reshape(nb * (ny + 1), ny)
    elif perturbed_p == 1:
        pE = pE + perturbation.reshape(nb * (ny + 1), ny)
    elif perturbed_p == 2:
        p2E = p2E + perturbation.reshape(nb * (ny + 1), ny)
    else:
        print("perturbed_p is not set corretly")


    # Compute gg for all perturbed inputs in one run
    ggE = G(xE, pE, p2E, t, [expand_num], auxG).view(nb, ny + 1, nx)

    # Extract the original and perturbed gg
    gg_original = ggE[:, 0, :]
    gg_perturbed = ggE[:, 1:, :]

    # Compute finite differences
    dGdx = ((gg_perturbed - gg_original.unsqueeze(1)) / epsilon).permute(0, 2, 1)

    return dGdx, gg_original




def finite_difference_jacobian_P(G, x, p, p2, t,  epsilon, auxG):
    nb, np = p.shape
    _, np2 = p2.shape
    _, nx = x.shape

    xE = torch.cat([x.repeat_interleave(np  + 1, dim=0),x.repeat_interleave(np2 , dim=0)], dim=0).double()
    pE = torch.cat([p.repeat_interleave(np  + 1, dim=0),p.repeat_interleave(np2, dim=0)], dim=0).double()
    p2E = torch.cat( [p2.repeat_interleave(np  + 1, dim=0),p2.repeat_interleave(np2, dim=0)], dim=0).double()


    perturbation_p = torch.eye(np).unsqueeze(0).expand(nb, -1, -1) * epsilon
    perturbation_p2 = torch.eye(np2).unsqueeze(0).expand(nb, -1, -1) * epsilon

    perturbation_p = torch.cat([torch.zeros(nb, 1, np), perturbation_p], dim=1).to(x)

    perturbation_p2 = perturbation_p2.to(x)



    pE[:nb * (np + 1),:] =  pE[:nb * (np + 1),:]  + perturbation_p.reshape(nb * (np + 1), np)
    p2E[nb * (np + 1):,:] = p2E[nb * (np + 1):,:] + perturbation_p2.reshape(nb * (np2), np2)
    # Expand x and add perturbations
   # xE = x.unsqueeze(1).expand(-1, np + 1, -1).reshape(nb * (np + 1), np) + perturbation.reshape(nb * (np + 1), np)

    # Compute gg for all perturbed inputs in one run
    ggE = G(xE, pE, p2E, t, [np + 1,np2], auxG)
    ggE_p = ggE[:nb * (np + 1),:].view(nb, np + 1, nx)
    ggE_p2 = ggE[nb * (np + 1):,:].view(nb, np2 , nx)

    # Extract the original and perturbed gg
    gg_original_p = ggE_p[:, 0, :]
    gg_perturbed_p = ggE_p[:, 1:, :]

    gg_perturbed_p2 = ggE_p2

    # Compute finite differences
    dGdp = ((gg_perturbed_p - gg_original_p.unsqueeze(1)) / epsilon).permute(0, 2, 1)
    dGdp2 = ((gg_perturbed_p2 - gg_original_p.unsqueeze(1)) / epsilon).permute(0, 2, 1)
    return dGdp, dGdp2



