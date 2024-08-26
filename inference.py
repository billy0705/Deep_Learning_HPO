from src.HPO_inference import HPOInference
import torch


model_save_path = "./model/HPO-model-weight-32.pth"
batch_size = 1
num_elements = 10  # Number of tuples in each set
dim_input = 17

HPO_inference = HPOInference(model_save_path)

c = torch.randn(batch_size, num_elements, dim_input, dtype=torch.float64)
x_hat = torch.randn(batch_size, 16, dtype=torch.float64)

x = HPO_inference.generator(x_hat, c)
print("x_denoised:", x)
