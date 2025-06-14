import torch
import matplotlib.pyplot as plt
import numpy as np
from scripts.models import *
from scripts.utils import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


u_model = VelocityNet().to(device)
P_model = PressureNet().to(device)
checkpoint_path = 'weights/model_0614_1.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
u_model.load_state_dict(checkpoint['u_model_state_dict'])
P_model.load_state_dict(checkpoint['P_model_state_dict'])

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
    vx_pred, vy_pred, p_pred = constraint_output(u_model, P_model, domain_tensor, domain_mask_vx, domain_mask_vy, domain_mask_p)
    p_pred_obs = constraint_output(u_model, P_model, domain_obs_tensor, obs_mask_vx, obs_mask_vy, obs_mask_p)[2]

scale = 0.01
vx_pred = scale * vx_pred.cpu().numpy()
vy_pred = scale * vy_pred.cpu().numpy()
p_pred = p_pred.cpu().numpy()
p_pred_obs = p_pred_obs.cpu().numpy()
p_obs = data_obs_np[:, 2:3]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Velocity Field (Quiver plot)
axs[0].quiver(domain_np[:, 0], domain_np[:, 1], vx_pred[:, 0], vy_pred[:, 0], scale=5)
axs[0].set_title("Predicted Velocity Field")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].axis('scaled')

# Pressure Field (Scatter plot with color)
sc = axs[1].scatter(domain_np[:, 0], domain_np[:, 1], c=p_pred[:, 0], cmap='viridis', s=1)
axs[1].set_title("Predicted Pressure Field")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")

axs[1].axis('scaled')
axs[1].scatter(domain_obs_np[:, 0], domain_obs_np[:, 1], c='red', marker='x', label='Observed P')
axs[1].scatter(domain_obs_np[:, 0], domain_obs_np[:, 1], c='white', edgecolors='black', marker='o', label='Predicted P')

for i in range(len(domain_obs_np)):
    x, y = domain_obs_np[i]
    pred_val = p_pred_obs[i, 0]
    obs_val = p_obs[i, 0]
    axs[1].text(x + 0.01, y + 0.01, f"{pred_val:.2f}/{obs_val:.2f}", fontsize=6, color='black')


axs[1].legend()

# Add colorbar for the pressure plot
fig.colorbar(sc, ax=axs[1], label="Pressure")

plt.tight_layout()
plt.savefig("prediction_results.png", dpi=300)
plt.show()