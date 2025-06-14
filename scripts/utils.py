import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def compute_masks(x):
    Lout = 1.5
    Lin = -0.5
    h = 0.5
    r = 0.075
    Ui = 5

    x0 = x[:, 0]
    x1 = x[:, 1]

    circle = 1- torch.exp(-((x0**2 + x1**2 - r**2)/0.01)**2)
    top = h - x1
    bottom = x1 + h
    wall_mask = (circle * top * bottom) / h**2
    
    left_r2 = torch.sqrt((Lin - x0)**2 + (x1)**2)
    inlet_mask = left_r2 / (left_r2 + 0.01)
    right = Lout - x0

    wall_mask = wall_mask.clamp(min=0.01)
    inlet_mask = inlet_mask.clamp(min=0.01)

    mask_vy = (wall_mask * inlet_mask).unsqueeze(1)
    mask_vx = [mask_vy, (Ui * (1 - inlet_mask) * wall_mask).unsqueeze(1)]
    mask_p = right.unsqueeze(1)

    # mask_vy = torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)
    # mask_vx = [mask_vy, torch.zeros((x.shape[0], 1), dtype=torch.float32, device=x.device)]
    # mask_p  = torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)

    return mask_vx, mask_vy, mask_p

def constraint_output(u_model, P_model, x, mask_vx, mask_vy, mask_p):
    vx, vy = u_model(x)
    P = P_model(x)

    vx = mask_vx[0]*vx + mask_vx[1]*P
    vy = mask_vy * vy
    P  = mask_p * P

    return vx, vy, P

##############################################################
def derivative(y, t):
    df = torch.autograd.grad(y, t, grad_outputs = torch.ones_like(y).to(device), create_graph = True)[0]
    df_x = df[:, 0:1]
    df_y = df[:, 1:2]
    return df_x, df_y

def requires_grad(x, device):
    return torch.tensor(x, dtype = torch.float32, requires_grad = True).to(device)

def PDE(u_model, P_model, domain, rho, vis, domain_mask_vx, domain_mask_vy, domain_mask_p):
    vx, vy, p = constraint_output(u_model, P_model, domain, domain_mask_vx, domain_mask_vy, domain_mask_p)

    dvx_x, dvx_y = derivative(vx, domain)
    dvx_xx, _ = derivative(dvx_x, domain)
    _, dvx_yy =  derivative(dvx_y, domain)

    dvy_x, dvy_y = derivative(vy, domain)
    dvy_xx, _ = derivative(dvy_x, domain)
    _, dvy_yy = derivative(dvy_y, domain)

    dp_x, dp_y = derivative(p, domain)

    pde_vx = rho * (vx * dvx_x + vy * dvx_y) + dp_x - vis * (dvx_xx + dvx_yy)
    pde_vy = rho * (vx * dvy_x + vy * dvy_y) + dp_y - vis * (dvy_xx + dvy_yy)
    pde_cont = dvx_x + dvy_y

    return pde_vx, pde_vy, pde_cont