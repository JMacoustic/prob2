import torch
import matplotlib.pyplot as plt
import numpy as np
from scripts.utils import *
from scripts.models import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

domain_path = "data/domain.csv"
domain_obs_path = "data/domain_obs.csv"
data_obs_path = "data/measurements_obs.csv"

domain = np.loadtxt(domain_path, delimiter=",", skiprows = 1)
domain_obs = np.loadtxt(domain_obs_path, delimiter=",", skiprows = 1)
data_obs = np.loadtxt(data_obs_path, delimiter=",", skiprows = 1)

# Precompute masks for domain
domain_tensor = torch.tensor(domain, dtype=torch.float32).to(device)
domain_mask_vx, domain_mask_vy, domain_mask_p = compute_masks(domain_tensor)

domain_obs_tensor = torch.tensor(domain_obs, dtype=torch.float32).to(device)
obs_mask_vx, obs_mask_vy, obs_mask_p = compute_masks(domain_obs_tensor)

dummy_u_model = DummyModel2()
dummy_P_model = DummyModel1()

# Get constraint output
vx_out, vy_out, P_out = constraint_output(dummy_u_model, dummy_P_model, domain_tensor, domain_mask_vx, domain_mask_vy, domain_mask_p)

# Convert to numpy for plotting
x = domain_tensor[:, 0].cpu().numpy()
y = domain_tensor[:, 1].cpu().numpy()
vx_np = vx_out[:, 0].cpu().numpy()
vy_np = vy_out[:, 0].cpu().numpy()
p_np  = P_out[:, 0].cpu().numpy()

# Plot velocity field
plt.figure(figsize=(6, 6))
plt.quiver(x, y, vx_np, vy_np, scale=5)
plt.title("Constrained Velocity Field")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.show()

# Optional: Pressure scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(x, y, c=p_np, cmap='coolwarm', s=5)
plt.title("Constrained Pressure Field")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.colorbar(label='Pressure')
plt.grid(True)
plt.show()