import torch
import wandb
import matplotlib.pyplot as plt
import numpy as np
import warnings
import random
import torch.nn as nn
import pandas as pd
from copy import deepcopy
from scripts.models import *

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

domain_path = "data/domain.csv"
domain_obs_path = "data/domain_obs.csv"
data_obs_path = "data/measurements_obs.csv"

domain = np.loadtxt(domain_path, delimiter=",", skiprows = 1)
domain_obs = np.loadtxt(domain_obs_path, delimiter=",", skiprows = 1)
data_obs = np.loadtxt(data_obs_path, delimiter=",", skiprows = 1)

###############boundary conditions###########################
Lout = 1.5
Lin = -0.5
h = 0.5
r = 0.075
Ui = 5

circle = domain[:, 0]**2 + domain[:, 1]**2  - r**2
top = h - domain[:, 1]
bottom = domain[:, 1] - (- h) 
left_y = domain[:, 0] - Lin
left_x = Ui + Lin - domain[:, 0]
right = Lout - domain[:, 0]

def compute_masks(x):
    x0 = x[:, 0]
    x1 = x[:, 1]
    
    circle = x0**2 + x1**2 - r**2
    top = h - x1
    bottom = x1 + h
    left_y = x0 - Lin
    left_x = Lin - x0
    right = Lout - x0

    mask_vx = (circle * top * bottom).float()
    mask_vy = (circle * top * bottom * left_y).float()
    mask_p  = right.float()
    
    return mask_vx, mask_vy, mask_p, left_x

# Precompute masks for domain
domain_tensor = torch.tensor(domain, dtype=torch.float32).to(device)
domain_mask_vx, domain_mask_vy, domain_mask_p, domain_left_x = compute_masks(domain_tensor)

domain_obs_tensor = torch.tensor(domain_obs, dtype=torch.float32).to(device)
obs_mask_vx, obs_mask_vy, obs_mask_p, obs_left_x = compute_masks(domain_obs_tensor)

def constraint_output(model, x, mask_vx=None, mask_vy=None, mask_p=None, left_x=None):
    vx, vy, P = model(x)

    if mask_vx is None or mask_vy is None or mask_p is None or left_x is None:
        mask_vx, mask_vy, mask_p, left_x = compute_masks(x)

    vx = mask_vx.unsqueeze(1) * (left_x * vx + Ui)
    vy = mask_vy.unsqueeze(1) * vy
    P  = mask_p.unsqueeze(1) * P

    return vx, vy, P

##############################################################

def derivative(y, t):
    df = torch.autograd.grad(y, t, grad_outputs = torch.ones_like(y).to(device), create_graph = True)[0]
    df_x = df[:, 0:1]
    df_y = df[:, 1:2]
    return df_x, df_y

def requires_grad(x):
    return torch.tensor(x, dtype = torch.float32, requires_grad = True).to(device)

def PDE(model, domain):
    vx, vy, p = constraint_output(model, domain, domain_mask_vx, domain_mask_vy, domain_mask_p, domain_left_x)

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


##############################################################

# Load checkpoint and set optimizer
wandb_name = "use_complex_pinn"

use_checkpoint = False
checkpoint_path = 'weights/model.pt'

rho_init = 100.
vis_init = 0.005

num_epochs = 50001
pde_weight = 1
data_weight = 5
lr_model = 5e-2
lr_rho = 1e-2
lr_vis = 1e-5
model = complexPINN().to(device)
loss_fn = nn.MSELoss()

##############################################################
if use_checkpoint:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    rho_init = checkpoint['density']
    vis_init = checkpoint['viscosity']

rho = torch.tensor(rho_init).to(device).requires_grad_(True)
vis = torch.tensor(vis_init).to(device).requires_grad_(True)

optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': lr_model},
    {'params': rho, 'lr': lr_rho},
    {'params': vis, 'lr': lr_vis}
])

if use_checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

best_loss = np.inf
loss_list = []
rho_list, vis_list = [], []

## Requires grad
domain = requires_grad(domain)
Y_obs = requires_grad(domain_obs)
data_obs = requires_grad(data_obs)

## wandb
wandb.init(project="pinn-fluid-inference", name=wandb_name, config={
    "epochs": num_epochs,
    "lr_model": lr_model,
    "lr_rho": lr_rho,
    "lr_vis": lr_vis,
    "pde_weight": pde_weight,
    "data_weight": data_weight,
    "rho_init": rho_init,
    "vis_init": vis_init,
    "use_checkpoint": use_checkpoint
})

epoch = 0
while epoch < num_epochs:
    ## PDE loss
    PDE_vx, PDE_vy, PDE_cont = PDE(model, domain)
    loss_PDE_vx = loss_fn(PDE_vx, torch.zeros_like(PDE_vx))
    loss_PDE_vy = loss_fn(PDE_vy, torch.zeros_like(PDE_vy))
    loss_PDE_cont = loss_fn(PDE_cont, torch.zeros_like(PDE_cont))
    loss_pde = loss_PDE_vx + loss_PDE_vy + loss_PDE_cont

    ## Data loss
    u_obs, v_obs, p_obs = constraint_output(model, Y_obs, obs_mask_vx, obs_mask_vy, obs_mask_p, obs_left_x)
    loss_data_u = loss_fn(u_obs, data_obs[:, 0:1])
    loss_data_v = loss_fn(v_obs, data_obs[:, 1:2])
    loss_data_p = loss_fn(p_obs, data_obs[:, 2:3])
    loss_data = loss_data_u + loss_data_v + loss_data_p

    loss = pde_weight*loss_pde + data_weight*loss_data

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    rho_list.append(rho.item())
    vis_list.append(vis.item())

    if epoch % 1000 == 0:
        print('EPOCH : %6d/%6d | Loss_PDE : %5.4f| Loss_DATA : %5.4f | RHO : %.4f | VIS : %.6f' \
                %(epoch, num_epochs, loss_pde.item(), loss_data.item(), rho.item(), vis.item()))

    wandb.log({
        "total_loss": loss.item(),
        "loss_pde": loss_pde.item(),
        "loss_data": loss_data.item(),
        "loss_pde_vx": loss_PDE_vx.item(),
        "loss_pde_vy": loss_PDE_vy.item(),
        "loss_pde_cont": loss_PDE_cont.item(),
        "loss_data_u": loss_data_u.item(),
        "loss_data_v": loss_data_v.item(),
        "loss_data_p": loss_data_p.item(),
        "rho": rho.item(),
        "viscosity": vis.item()
    }, step=epoch)

    epoch += 1



##############################################################

# with torch.no_grad():
#     vx_pred, vy_pred, p_pred = model(domain)

# vx_pred = vx_pred.cpu().numpy()
# vy_pred = vy_pred.cpu().numpy()
# p_pred = p_pred.cpu().numpy()
# domain_np = domain.detach().cpu().numpy()

# plt.figure(figsize=(6, 6))
# plt.quiver(domain_np[:, 0], domain_np[:, 1], vx_pred[:, 0], vy_pred[:, 0], scale=5)
# plt.title("Predicted Velocity Field")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis('scaled')
# plt.show()

# plt.figure(figsize=(6, 6))
# sc = plt.scatter(domain_np[:, 0], domain_np[:, 1], c=p_pred[:, 0], cmap='viridis', s=5)
# plt.colorbar(sc, label="Pressure")
# plt.title("Predicted Pressure Field")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.axis('scaled')
# plt.show()

print(f"\nFinal learned rho: {rho.item():.4f}")
print(f"Final learned viscosity: {vis.item():.6f}")

save = True
if save:
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimal_state_dict': deepcopy(model.state_dict()),
                'loss': loss_list,
                'density': rho.item(),
                'viscosity': vis.item(),
                }, 'model.pt')
    
    artifact = wandb.Artifact('pinn-model', type='model')
    artifact.add_file('model.pt')
    wandb.log_artifact(artifact)

    print("model saved")

wandb.finish()