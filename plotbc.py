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
scale = 0.1
x = domain_tensor[:, 0].cpu().numpy()
y = domain_tensor[:, 1].cpu().numpy()
vx_np = scale * vx_out[:, 0].cpu().numpy()
vy_np = scale * vy_out[:, 0].cpu().numpy()
p_np  = P_out[:, 0].cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Velocity field
axes[0].quiver(x, y, vx_np, vy_np, scale=5)
axes[0].set_title("Constrained Velocity Field")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].axis('equal')
axes[0].grid(True)

# Pressure field
sc = axes[1].scatter(x, y, c=p_np, cmap='coolwarm', s=5)
axes[1].set_title("Constrained Pressure Field")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].axis('equal')
axes[1].grid(True)

# Add colorbar
cbar = fig.colorbar(sc, ax=axes[1], orientation='vertical', label='Pressure')

plt.tight_layout()
plt.show()