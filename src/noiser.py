import numpy as np


class NoiseAdder:

    def __init__(self, beta_start, beta_end, beta_timesteps,
                 schedule_method: str) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_method = schedule_method
        self.timesteps = beta_timesteps
        self.beta_schedule = self.beta_scheduler()
        # print(f'{self.beta_schedule}')
        # calulatinf alpha = 1 - beta
        self.alpha_schedule = 1 - self.beta_schedule
        # print(f'{self.beta_schedule=}')
        # self.t = t  # Timesteps for dataset
        # self.x = x #dataset

    def add_noise(self, x, t):
        # print('Adding noise ')
        # total product of alpha schedule till t
        alpha_cumprod_t = np.prod(self.alpha_schedule[:t])
        # generates random gausian noise
        noise = np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        # calculating x_hat
        x_hat = (np.sqrt(alpha_cumprod_t) * x + np.sqrt(1 - alpha_cumprod_t) *
                 noise)
        return x_hat, noise

    def beta_scheduler(self):
        # print(f'{self.schedule_method=}')
        if self.schedule_method == "linear":
            # Returns evenly spaced betas over a specified timesteps
            beta_schedule = np.linspace(self.beta_start, self.beta_end,
                                        self.timesteps)
        elif self.schedule_method == "quadratic":
            beta_schedule = ((np.linspace(0, 1, self.timesteps) ** 2) *
                             (self.beta_end - self.beta_start) +
                             beta_start)
        elif self.schedule_method == "cosine":
            beta_schedule = (np.cos(np.linspace(0, np.pi / 2, self.timesteps)) *
                             (self.beta_start - self.beta_end) + self.beta_end)
        return beta_schedule


if __name__ == "__main__":
    beta_timesteps = 100  # Timesteps for beta scheduler
    beta_start = 0.001
    beta_end = 0.05
    schedule_method = 'cosine'  # may be 'linear', 'cosine' or 'quadraric'
    x = np.random.uniform(0, 1, size=(10, 16))

    t = 100  # timesteps

    noise_adder = NoiseAdder(beta_start, beta_end,  beta_timesteps,
                             schedule_method, x, t)

    x_hat, final_noise = noise_adder.add_noise()

    print("Original Configuration x:", x)
    print("Noisy Configuration x_hat:", x_hat)
    print("Added noise: ", final_noise)
