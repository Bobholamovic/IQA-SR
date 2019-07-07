import torch
import torch.nn.functional as F_

def MDSI(ref, dist, comb_method='sum'):
    # Pytorch version of MDSI
    assert comb_method in ('sum', 'mult')
    
    _D = ref.device
    
    C1 = 140; 
    C2 = 55; 
    C3 = 550;
    dx = (torch.Tensor([1, 0, -1])/3).repeat(1, 1, 3, 1).to(_D)
    dy = dx.transpose(-1, -2)
    
    rows, cols = ref.size()[-2:]
    min_dimension = min(rows, cols)
    f = max(1, round(min_dimension / 256))
    ave_kernel = (torch.ones(3,1,f,f)/(f*f)).to(_D)
    
    if f % 2 == 0:
        ave_ref = F_.pad(ref.float(), (f//2-1, f//2, f//2-1, f//2))
        ave_ref = F_.conv2d(ave_ref, ave_kernel, bias=None, stride=1, groups=3)
    else:
        ave_ref = F_.conv2d(ref.float(), ave_kernel, bias=None, stride=1, padding=f//2, groups=3)
    ave_ref = ave_ref[...,::f, ::f]
    if f % 2 == 0:
        ave_dist = F_.pad(dist.float(), (f//2-1, f//2, f//2-1, f//2))
        ave_dist = F_.conv2d(ave_dist, ave_kernel, bias=None, stride=1, groups=3)
    else:
        ave_dist = F_.conv2d(dist.float(), ave_kernel, bias=None, stride=1, padding=f//2, groups=3)
    ave_dist = ave_dist[...,::f, ::f]
    
    WL = torch.Tensor([0.2989, 0.5870, 0.1140]).view(1,3,1,1).to(_D)
    WH = torch.Tensor([0.30, 0.04, -0.35]).view(1,3,1,1).to(_D)
    WM = torch.Tensor([0.34, -0.60, 0.17]).view(1,3,1,1).to(_D)
    
    L1 = (ave_ref*WL).sum(1, keepdim=True)
    L2 = (ave_dist*WL).sum(1, keepdim=True)
    F = 0.5 * (L1 + L2)
    
    
    H1 = (ave_ref*WH).sum(1, keepdim=True)
    H2 = (ave_dist*WH).sum(1, keepdim=True)
    M1 = (ave_ref*WM).sum(1, keepdim=True)
    M2 = (ave_dist*WM).sum(1, keepdim=True)   
    
    ix_L1 = F_.conv2d(L1, dx, bias=None, stride=1, padding=(1,1))
    iy_L1 = F_.conv2d(L1, dy, bias=None, stride=1, padding=(1,1))
    gR = torch.sqrt(ix_L1**2 + iy_L1**2)
    
    ix_L2 = F_.conv2d(L2, dx, bias=None, stride=1, padding=(1,1))
    iy_L2 = F_.conv2d(L2, dy, bias=None, stride=1, padding=(1,1))
    gD = torch.sqrt(ix_L2**2 + iy_L2**2)
    
    ix_F = F_.conv2d(F, dx, bias=None, stride=1, padding=(1,1))
    iy_F = F_.conv2d(F, dy, bias=None, stride=1, padding=(1,1))
    gF = torch.sqrt(ix_F**2 + iy_F**2)
    
    GS12 = (2 * gR * gD + C1) / (gR ** 2 + gD ** 2 + C1)
    GS13 = (2 * gR * gF + C2) / (gR ** 2 + gF ** 2 + C2)
    GS23 = (2 * gD * gF + C2) / (gD ** 2 + gF ** 2 + C2)
    GS_HVS = GS12 + GS23 - GS13
    
    CS = (2 * (H1 * H2 + M1 * M2) + C3) / (H1 ** 2 + H2 ** 2 + M1 ** 2 + M2 ** 2 + C3)
    
    if comb_method == 'sum':
        alpha = 0.6
        GCS = alpha * GS_HVS + (1 - alpha) * CS
    elif comb_method == 'mult':
        gamma = 0.2
        beta = 0.1
        GCS = GS_HVS ** gamma * CS ** beta
    
    GCS = (F_.relu(GCS.view(-1)) ** 0.5) ** 0.5 
    Q = torch.mean(torch.abs(GCS-GCS.mean())) ** 0.25
    
    return Q
