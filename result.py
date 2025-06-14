import torch
import matplotlib.pyplot as plt
import numpy as np
from scripts.models import *
from scripts.utils import *
import pandas as pd
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


u_model = VelocityNet().to(device)
P_model = PressureNet().to(device)
checkpoint_path = 'weights/model_0614_9.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
u_model.load_state_dict(checkpoint['u_model_state_dict'])
P_model.load_state_dict(checkpoint['P_model_state_dict'])
rho = checkpoint['density']
vis = checkpoint['viscosity']

domain_path = "data/domain.csv"
domain_obs_path = "data/domain_obs.csv"
data_obs_path = "data/measurements_obs.csv"

domain_np = np.loadtxt(domain_path, delimiter=",", skiprows = 1)
domain_obs_np = np.loadtxt(domain_obs_path, delimiter=",", skiprows = 1)
data_obs_np = np.loadtxt(data_obs_path, delimiter=",", skiprows = 1)

# Precompute masks for domain
domain_tensor = torch.tensor(domain_np, dtype=torch.float32).to(device)
domain_mask_vx, domain_mask_vy, domain_mask_p = compute_masks(domain_tensor)

domain_obs_tensor = torch.tensor(domain_obs_np, dtype=torch.float32).to(device)
obs_mask_vx, obs_mask_vy, obs_mask_p = compute_masks(domain_obs_tensor)


with torch.no_grad():
    vx, vy, p = constraint_output(u_model, P_model, domain_tensor, domain_mask_vx, domain_mask_vy, domain_mask_p)


vx = vx.detach().cpu().numpy() #(4711,1)
vy = vy.detach().cpu().numpy() #(4711,1)
p  = p.detach().cpu().numpy()  #(4711,1)


rows = []

for i in range(vx.shape[0]):
    rows.append([i + 1, p[i].item(), vx[i].item(), vy[i].item(), rho, vis])

columns = ['ID', 'pressure', 'x-velocity', 'y-velocity', 'rho', 'mu']
df = pd.DataFrame(rows, columns=columns)

output_path = "./data.csv"
df.to_csv(output_path, index=False)