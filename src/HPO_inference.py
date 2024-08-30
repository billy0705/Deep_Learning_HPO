import torch
from src.HPO_network import HPONetwork
from src.denoiser import Denoiser


class HPOInference():
    def __init__(self, model_save_path, dim_input=17, num_outputs=32,
                 dim_output=16, device="cpu", beta_timesteps=1001,
                 beta_start=0.001, beta_end=0.05, schedule_method='cosine'):

        self.model_save_path = model_save_path
        self.dim_input = dim_input
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.device = device
        self.beta_timesteps = beta_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_method = schedule_method

        self.get_device()
        self.init_model()
        self.init_denoiser()

    def get_device(self):
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        self.device = torch.device(dev)

    def init_model(self):
        self.model = HPONetwork(self.dim_input, self.num_outputs,
                                self.dim_output)
        self.model.to(self.device)
        self.model = torch.load(self.model_save_path)
        self.model = self.model.to(torch.float64)

    def init_denoiser(self):
        self.denoiser = Denoiser(self.beta_start, self.beta_end,
                                 self.beta_timesteps, self.schedule_method)

    def generator(self, x_hat, c):
        t = 1000  # Timestep
        improve = torch.ones(1, dtype=torch.float64)
        x_hat = x_hat.to(self.device)
        improve = improve.to(self.device)
        c = c.to(self.device)
        noise_pred = self.model(x_hat, improve, c)
        # print(f"{noise_pred=}")
        x_denoised = self.denoiser.denoise(x_hat, noise_pred, t)

        # print("Noisy Configuration x_hat:", x_hat)
        # print("Reconstructed Configuration x_reconstructed:", x_denoised)
        return x_denoised
