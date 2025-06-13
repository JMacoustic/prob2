import torch
import matplotlib.pyplot as plt
import numpy as np

domain_path = "data/domain.csv"
domain_obs_path = "data/domain_obs.csv"
data_obs_path = "data/measurements_obs.csv"

domain = np.loadtxt(domain_path, delimiter=",", skiprows = 1)
domain_obs = np.loadtxt(domain_obs_path, delimiter=",", skiprows = 1)
data_obs = np.loadtxt(data_obs_path, delimiter=",", skiprows = 1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

###############boundary conditions###########################
Lout = 1.5
Lin = -0.5
h = 0.5
r = 0.075
Ui = 5

def compute_masks(x):
    x0 = x[:, 0]
    x1 = x[:, 1]

    circle = x0**2 + x1**2 - r**2
    top = h - x1
    bottom = x1 + h
    left_y = x0 - Lin
    left_x = Lin - x0
    right = Lout - x0

    return (
        (circle * top * bottom).unsqueeze(1),                    # mask_vx
        (circle * top * bottom * left_y).unsqueeze(1),          # mask_vy
        right.unsqueeze(1),                                     # mask_p
        left_x.unsqueeze(1),                                    # left_x
    )

# Precompute masks for domain
domain_tensor = torch.tensor(domain, dtype=torch.float32).to(device)
domain_mask_vx, domain_mask_vy, domain_mask_p, domain_left_x = compute_masks(domain_tensor)

domain_obs_tensor = torch.tensor(domain_obs, dtype=torch.float32).to(device)
obs_mask_vx, obs_mask_vy, obs_mask_p, obs_left_x = compute_masks(domain_obs_tensor)

def constraint_output(model, x, mask_vx, mask_vy, mask_p, left_x):
    vx, vy, P = model(x)

    vx = mask_vx * (left_x * vx + Ui)
    vy = mask_vy * vy
    P  = mask_p * P

    return vx, vy, P

# Dummy model that outputs all ones
class DummyModel(torch.nn.Module):
    def forward(self, x):
        N = x.shape[0]
        return torch.tensor((N, 1), device=x.device), torch.ones((N, 1), device=x.device), torch.ones((N, 1), device=x.device)

dummy_model = DummyModel()

# Get constraint output
vx_out, vy_out, P_out = constraint_output(dummy_model, domain_tensor, domain_mask_vx, domain_mask_vy, domain_mask_p, domain_left_x)

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