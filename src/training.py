from torch import no_grad


def train_loop(dataloader, model, loss_fn, optimizer, device, batch_size):
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
        x_bar = x_bar.to(device)
        improve = improve.to(device)
        c = c.to(device)
        e = e.to(device)
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


def test_loop(dataloader, model, loss_fn, device):
    model.eval()
    # size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0
    # correct = 0

    with no_grad():
        for x_bar, e, improve, c in dataloader:
            x_bar = x_bar.to(device)
            improve = improve.to(device)
            c = c.to(device)
            e = e.to(device)
            pred = model(x_bar, improve, c)
            test_loss += loss_fn(pred, e).item()
            # correct += (pred.argmax(1) == e).type(torch.float).sum().item()

    test_loss /= num_batches
    # correct /= size
    print(f"""Test  Avg loss: {test_loss:>8f}""")
    return test_loss
