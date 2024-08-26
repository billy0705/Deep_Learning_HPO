from src.HPO_network import HPONetwork
from src.denoiser import Denoiser
import torch


dim_input = 17  # x + y
num_outputs = 32
dim_output = 16

model_save_path = "./model/HPO-model-weight-32.pth"

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
model = HPONetwork(dim_input, num_outputs, dim_output)
model.to(device)
model = torch.load(model_save_path)
model = model.to(torch.float64)


batch_size = 1
num_elements = 10  # Number of tuples in each set

c = torch.randn(batch_size, num_elements, dim_input, dtype=torch.float64)
x_hat = torch.randn(batch_size, 16, dtype=torch.float64)
improve = torch.ones(1, dtype=torch.float64)

print("c Shape:", c.shape)
print("x_hat Shape:", x_hat.shape)
print("improve Shape:", improve.shape)


x_hat = x_hat.to(device)
improve = improve.to(device)
c = c.to(device)
noise_pred = model(x_hat, improve, c)

print(f"{noise_pred=}")

beta_timesteps = 100  # Timesteps for beta scheduler
beta_start = 0.001
beta_end = 0.05
schedule_method = 'cosine'  # may be 'linear', 'cosine' or 'quadraric'
# Assume we have a noisy configuration
# x_hat = np.random.uniform(0, 1, size=(10, 16))

t = 50  # Timestep
# Gaussian noise
# noise = np.random.normal(loc=0.0, scale=1.0, size=x_hat.shape)
denoiser = Denoiser(beta_start, beta_end, beta_timesteps, schedule_method)

x_reconstructed = denoiser.denoise(x_hat, noise_pred, t)

print("Noisy Configuration x_hat:", x_hat)
print("Reconstructed Configuration x_reconstructed:", x_reconstructed)
