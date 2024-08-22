import numpy as np

class NoiseAdder:

    def __init__(self, beta, beta_timesteps, x, t) -> None:
        
        self.beta_schedule = np.linspace(beta, 0.05, beta_timesteps) #Returns evenly spaced betas over a specified timesteps
        self.alpha_schedule = 1 - self.beta_schedule  # calulatinf alpha = 1 - beta 
        self.alpha_com_prod = np.prod(self.alpha_schedule) # a = total product of alpha schedule
        self.x = x
        self.t = t # timestepd for dataset

    def add_noise(self):
        print('Adding noise ')
         # Compute the cumulative product of alpha up to time t
        alpha_cumprod_t = self.alpha_com_prod
        noise = np.random.normal(loc=0.0, scale=1.0, size=x.shape) # generates random gausian noise
        x_hat = np.sqrt(alpha_cumprod_t) * self.x + np.sqrt(1 - alpha_cumprod_t) * noise # calculating x_hat using std ddpm 

        return x_hat

if __name__ == "__main__":
    beta_timesteps = 100 # timesteps for beta scheduler
    beta = 0.001
    x = np.random.uniform(0, 1, size=(10,10))

    t = 100 # timesteps

    noise_adder = NoiseAdder(beta, beta_timesteps, x , t)

    x_hat = noise_adder.add_noise()

    print("Original Configuration x:", x)
    print("Noisy Configuration x_hat:", x_hat)
    


