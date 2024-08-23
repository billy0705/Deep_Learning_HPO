from HPO_dataset import HPODataset
from torch.utils.data import DataLoader
from model.HPO_network import HPONetwork
from torch import nn, optim, no_grad
# import torch


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for
    # batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (x_bar, improve, c, e) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(x_bar, improve, c)
        loss = loss_fn(pred, e)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(x_bar)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with no_grad():
        for x_bar, improve, c, e in dataloader:
            pred = model(x_bar, improve, c)
            test_loss += loss_fn(pred, e).item()
            # correct += (pred.argmax(1) == e).type(torc).sum().item()

    test_loss /= num_batches
    # correct /= size
    # print(f"Test Error:\n Accuracy:
    # {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")


learning_rate = 1e-3
batch_size = 64
# epochs = 5
dim_input = 17  # x + y
num_outputs = 32
dim_output = 16

training_data = HPODataset("./data/train.json")
test_data = HPODataset("./data/test.json")
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model = HPONetwork(dim_input, num_outputs, dim_output)
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
