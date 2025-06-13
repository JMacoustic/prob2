import torch
import matplotlib.pyplot as plt
import numpy as np
from scripts.models import *
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = PINN().to(device)
checkpoint_path = 'weights/model_1.pt'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

domain_path = "data/domain.csv"
domain_obs_path = "data/domain_obs.csv"
data_obs_path = "data/measurements_obs.csv"

domain_np = np.loadtxt(domain_path, delimiter=",", skiprows = 1)
domain = torch.tensor(domain_np, dtype=torch.float32).to(device)

with torch.no_grad():
    vx_pred, vy_pred, p_pred = model(domain)

vx_pred = vx_pred.cpu().numpy()
vy_pred = vy_pred.cpu().numpy()
p_pred = p_pred.cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Velocity Field (Quiver plot)
axs[0].quiver(domain_np[:, 0], domain_np[:, 1], vx_pred[:, 0], vy_pred[:, 0], scale=5)
axs[0].set_title("Predicted Velocity Field")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
axs[0].axis('scaled')

# Pressure Field (Scatter plot with color)
sc = axs[1].scatter(domain_np[:, 0], domain_np[:, 1], c=p_pred[:, 0], cmap='viridis', s=5)
axs[1].set_title("Predicted Pressure Field")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
axs[1].axis('scaled')

# Add colorbar for the pressure plot
fig.colorbar(sc, ax=axs[1], label="Pressure")

plt.tight_layout()
plt.show()