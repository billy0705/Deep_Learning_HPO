import numpy as np

class Denoiser:

    def __init__(self, beta, beta_timesteps, x_hat, noise, t) -> None:
        self.beta = beta
        self.beta_schedule = np.linspace(beta, 0.05, beta_timesteps)  # Returns evenly spaced betas over a specified timesteps
        self.alpha_schedule = 1 - self.beta_schedule  # alpha = 1 - beta
        self.alpha_com_prod = np.prod(self.alpha_schedule)  # product of alphas
        self.x_hat = x_hat
        self.noise = noise
        self.t = t  # Timesteps for dataset

    def denoise(self):
        print('Denoising X_hat')
        alpha_cumprod_t = self.alpha_com_prod # product of all alphas
        print(f'{alpha_cumprod_t = }')
        noise = np.random.normal(loc=0.0, scale=1.0, size=self.x_hat.shape)  # Gaussian noise 
        print(f'{noise = }')
        beta_1 = self.beta_schedule[1]
        alpha_1 = self.alpha_schedule[1]
        print(f'{beta_1 = }')
        print(f'{alpha_1 = }')

        # reconstructing x from x_hat given noise
        x_reconstructed = (1/np.sqrt(1 - beta_1))*(self.x_hat - ((self.beta/np.sqrt(1-alpha_1))*noise))

        return x_reconstructed

if __name__ == "__main__":
    beta_timesteps = 100  # Timesteps for beta scheduler
    beta = 0.001
    noise = 0
    x_hat = np.random.uniform(0, 1, size=(10,10))  # Assume we have a noisy configuration

    t = 100  # Timesteps

    denoiser = Denoiser(beta, beta_timesteps, x_hat, noise, t)

    x_reconstructed = denoiser.denoise()

    print("Noisy Configuration x_hat:", x_hat)
    print("Reconstructed Configuration x_reconstructed:", x_reconstructed)
