from src.HPO_dataset import HPODataset
from torch.utils.data import DataLoader
from src.HPO_network import HPONetwork
from src.training import train_loop, test_loop
from torch import nn, optim
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random

epochs = 100
learning_rate = 1e-3
batch_size = 128
dim_input = 17  # x + y
num_outputs = 32
dim_output = 16
model_save_path = "./model/HPO-model-weight-32-0831.pth"
if not os.path.exists("./model/"):
    os.mkdir("./model/")

training_data = HPODataset("./data/train.json")
test_data = HPODataset("./data/test.json")
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
model = HPONetwork(dim_input, num_outputs, dim_output)
model.to(device)

# model = torch.load(model_save_path)

model = model.to(torch.float64)
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.1, patience=10)

train_history = []
test_history = []
min_test_loss = 1000
for t in range(epochs):
    print("-------------------------------")
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(train_dataloader, model, loss_fn,
                            optimizer, device, batch_size)
    test_loss = test_loop(test_dataloader, model, loss_fn, device)
    train_history.append(train_loss)
    test_history.append(test_loss)
    current_lr = float(optimizer.param_groups[0]["lr"])
    num_samples = random.randint(1, 50)
    train_dataloader.dataset.set_number_sample(num_samples)
    test_dataloader.dataset.set_number_sample(num_samples)
    print(f"Current sample number: {num_samples}")
    print(f"Current learning rate: {current_lr:.6f}")
    if t == 0 or test_history[t] < min_test_loss:
        print("Saving model")
        min_test_loss = test_history[t]
        torch.save(model, model_save_path)
    scheduler.step(test_loss)


print("-------------------------------")

if not os.path.exists("./plot/"):
    os.mkdir("./plot/")

x = np.arange(1, len(train_history)+1)

print(x)
plt.clf()
plt.plot(x, train_history, label="train_history")
plt.plot(x, test_history, label="test_history")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.title("Loss History")
plt.savefig("./plot/loss-32.png")
print("Done!")
