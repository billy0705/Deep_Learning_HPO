import numpy as np


class Denoiser:

    def __init__(self, beta_start, beta_end, beta_timesteps,
                 schedule_method: str) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_method = schedule_method
        self.timesteps = beta_timesteps
        # Returns evenly spaced betas over a specified timesteps
        self.beta_schedule = self.beta_scheduler()
        self.alpha_schedule = 1 - self.beta_schedule  # alpha = 1 - beta
        # self.x_hat = x_hat
        # self.noise = noise
        # self.t = t  # Timesteps for dataset

        print(f'{self.schedule_method=}')

    def denoise(self, x_hat, noise, t):
        print('Denoising X_hat')
        # product of all alphas
        alpha_cumprod_t = np.prod(self.alpha_schedule[:t])
        print(f'{alpha_cumprod_t=}')
        print(f'{noise=}')
        alpha_t = self.alpha_schedule[t]
        print(f'{alpha_t=}')

        # reconstructing x from x_hat given noise
        x_reconstructed = ((1/np.sqrt(alpha_t)) *
                           (x_hat - (((1-alpha_t) /
                                      np.sqrt(1-alpha_cumprod_t))*noise)))

        return x_reconstructed

    def beta_scheduler(self):

        if self.schedule_method == "linear":
            # Returns evenly spaced betas over a specified timesteps
            beta_schedule = np.linspace(self.beta_start, self.beta_end,
                                        beta_timesteps)
        elif self.schedule_method == "quadratic":
            beta_schedule = ((np.linspace(0, 1, self.timesteps) ** 2) *
                             (self.beta_end - self.beta_start) + beta_start)
        elif self.schedule_method == "cosine":
            beta_schedule = (np.cos(np.linspace(0, np.pi / 2, self.timesteps)) *
                             (self.beta_start - self.beta_end) + self.beta_end)
        return beta_schedule


if __name__ == "__main__":
    beta_timesteps = 100  # Timesteps for beta scheduler
    beta_start = 0.001
    beta_end = 0.05
    noise = 0
    schedule_method = 'cosine'  # may be 'linear', 'cosine' or 'quadraric'
    # Assume we have a noisy configuration
    x_hat = np.random.uniform(0, 1, size=(10, 16))

    t = 50  # Timestep
    # Gaussian noise
    noise = np.random.normal(loc=0.0, scale=1.0, size=x_hat.shape)
    denoiser = Denoiser(beta_start, beta_end, beta_timesteps, schedule_method)

    x_reconstructed = denoiser.denoise(x_hat, noise, t)

    print("Noisy Configuration x_hat:", x_hat)
    print("Reconstructed Configuration x_reconstructed:", x_reconstructed)
