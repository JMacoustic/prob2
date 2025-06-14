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
from scripts.utils import *
import json

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

############### domain setup and boundary conditions###########################

domain_path = "data/domain.csv"
domain_obs_path = "data/domain_obs.csv"
data_obs_path = "data/measurements_obs.csv"

domain_np = np.loadtxt(domain_path, delimiter=",", skiprows = 1)
domain_obs_np = np.loadtxt(domain_obs_path, delimiter=",", skiprows = 1)
data_obs_np = np.loadtxt(data_obs_path, delimiter=",", skiprows = 1)

domain_torch = torch.tensor(domain_np, dtype=torch.float32).to(device)
domain_obs_torch = torch.tensor(domain_obs_np, dtype=torch.float32).to(device)
data_obs_torch = torch.tensor(data_obs_np, dtype=torch.float32).to(device)

# pre compute mask tensor 
domain_mask_vx, domain_mask_vy, domain_mask_p = compute_masks(domain_torch)
obs_mask_vx, obs_mask_vy, obs_mask_p = compute_masks(domain_obs_torch)

############################ hyperparameters setup ##################################
with open("scripts/config.json", "r") as f:
    config = json.load(f)

wandb_name = config["wandb_name"]
use_checkpoint = config["use_checkpoint"]
checkpoint_path = config["checkpoint_path"]
rho_init = config["rho_init"]
vis_init = config["vis_init"]
num_epochs = config["num_epochs"]
pde_weight = config["pde_weight"]  # list[vx, vy, cont]
data_weight = config["data_weight"]  # list[vx, vy, P]
bc_weight = config["bc_weight"]  # float
lr_u = config["lr_u"]
lr_P = config["lr_P"]
lr_rho = config["lr_rho"]
lr_vis = config["lr_vis"]
u_model = VelocityNet().to(device)
P_model = PressureNet().to(device)
loss_fn = nn.MSELoss()

if use_checkpoint:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    rho_init = checkpoint['density']
    vis_init = checkpoint['viscosity']

rho = torch.tensor(rho_init).to(device).requires_grad_(True)
vis = torch.tensor(vis_init).to(device).requires_grad_(True)
# domain = domain_tensor.requires_grad_(True)
# Y_obs = domain_obs_tensor.requires_grad_(True)
# data_obs = data_obs_tensor.requires_grad_(True)

optimizer = torch.optim.Adam([
    {'params': u_model.parameters(), 'lr': lr_u},
    {'params': P_model.parameters(), 'lr': lr_P},
    {'params': rho, 'lr': lr_rho},
    {'params': vis, 'lr': lr_vis}
])

if use_checkpoint:
    u_model.load_state_dict(checkpoint['u_model_state_dict'])
    P_model.load_state_dict(checkpoint['P_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


##################################### training setup ########################################
best_loss = np.inf
loss_list = []
rho_list, vis_list = [], []
epoch = 0

domain = requires_grad(domain_torch, device)
domain_obs = requires_grad(domain_obs_torch, device)
data_obs = requires_grad(data_obs_torch, device)

############################# training loop ##################################################
wandb.init(project="pinn-fluid-inference", name=wandb_name, config=config)
while epoch < num_epochs:
    ## PDE loss
    PDE_vx, PDE_vy, PDE_cont, u_pred, v_pred, P_pred = PDE(u_model, P_model, domain, rho, vis, domain_mask_vx, domain_mask_vy, domain_mask_p)
    loss_PDE_vx = loss_fn(PDE_vx, torch.zeros_like(PDE_vx))
    loss_PDE_vy = loss_fn(PDE_vy, torch.zeros_like(PDE_vy))
    loss_PDE_cont = loss_fn(PDE_cont, torch.zeros_like(PDE_cont))
    loss_pde = pde_weight[0]*loss_PDE_vx + pde_weight[1]*loss_PDE_vy + pde_weight[2]*loss_PDE_cont

    ## Data loss
    u_obs, v_obs, p_obs = constraint_output(u_model, P_model, domain_obs, obs_mask_vx, obs_mask_vy, obs_mask_p)
    loss_data_u = loss_fn(u_obs, data_obs[:, 0:1])
    loss_data_v = loss_fn(v_obs, data_obs[:, 1:2])
    loss_data_p = loss_fn(p_obs, data_obs[:, 2:3])
    loss_data = data_weight[0]*loss_data_u + data_weight[1]*loss_data_v + data_weight[2]*loss_data_p

    ## Boundary Loss
    loss_BC = bc_weight * circle_loss(u_pred, v_pred, domain)

    loss = loss_pde + loss_data + loss_BC

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_list.append(loss.item())
    rho_list.append(rho.item())
    vis_list.append(vis.item())

    if epoch % 1000 == 0:
        print('EPOCH : %6d/%6d | Loss_PDE : %5.4f| Loss_DATA : %5.4f | Loss_BC : %5.4f | RHO : %.4f | VIS : %.6f' \
                %(epoch, num_epochs, loss_pde.item(), loss_data.item(), loss_BC.item(), rho.item(), vis.item()))
    if epoch > 200:
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



############################### learned result #####################################

print(f"\nFinal learned rho: {rho.item():.4f}")
print(f"Final learned viscosity: {vis.item():.6f}")

save = True
if save:
    torch.save({
                'epoch': epoch,
                'u_model_state_dict': u_model.state_dict(),
                'P_model_state_dict': P_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_list,
                'density': rho.item(),
                'viscosity': vis.item(),
                }, 'model.pt')
    
    artifact = wandb.Artifact('pinn-model', type='model')
    artifact.add_file('model.pt')
    wandb.log_artifact(artifact)

    print("model saved")

wandb.finish()