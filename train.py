from src.HPO_dataset import HPODataset
from torch.utils.data import DataLoader
from src.HPO_network import HPONetwork
from torch import nn, optim, no_grad
import torch
import os


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    print_interval = int(0.3 * size / batch_size)
    if print_interval == 0:
        print_interval = 1
    # Set the model to training mode - important for
    # batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    batch = 0
    total_loss = 0
    for x_bar, e, improve, c in dataloader:
        # Compute prediction and loss
        pred = model(x_bar, improve, c)
        loss = loss_fn(pred, e)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % print_interval == 0:
            loss_num, current = loss.item(), batch * batch_size + len(x_bar)
            print(f"loss: {loss_num:>7f}  [{current:>5d}/{size:>5d}]")
        total_loss += loss.item()
        batch += 1

    avg_loss, current = total_loss/batch, batch * batch_size + len(x_bar)
    # print(f"Train Avg loss: {avg_loss:>7f}  [{current:>5d}/{size:>5d}]")
    print(f"Train Avg loss: {avg_loss:>7f}")
    return avg_loss


def test_loop(dataloader, model, loss_fn):
    model.eval()
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    # correct = 0

    with no_grad():
        for x_bar, e, improve, c in dataloader:
            pred = model(x_bar, improve, c)
            test_loss += loss_fn(pred, e).item()
            # correct += (pred.argmax(1) == e).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    print(f"""Test  Avg loss: {test_loss:>8f}""")
    return test_loss


epochs = 10
learning_rate = 1e-3
batch_size = 128
dim_input = 17  # x + y
num_outputs = 32
dim_output = 16
model_save_path = "./model/HPO-model-weight.pth"
if not os.path.exists("./model/"):
    os.mkdir("./model/")

training_data = HPODataset("./data/train.json")
test_data = HPODataset("./data/test.json")
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = HPONetwork(dim_input, num_outputs, dim_output)
model = model.to(torch.float64)
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

train_history = []
test_history = []
for t in range(epochs):
    print("-------------------------------")
    print(f"Epoch {t+1}\n-------------------------------")
    train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loss = test_loop(test_dataloader, model, loss_fn)
    train_history.append(train_loss)
    test_history.append(test_loss)
    if t == 0 or test_history[t] < test_history[t-1]:
        print("Saving model")
        torch.save(model, model_save_path)


print("-------------------------------")
print("Done!")
