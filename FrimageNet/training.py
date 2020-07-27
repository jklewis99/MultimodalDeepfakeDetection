import torch
from frimagenet_dataset import FrimagenetDataset
from frimagenet import FrimageNet
from torch.utils.data import DataLoader

def train(model, spectrogram_folder, xception_folder, loss_function, optimizer, epochs=100, batch_size=5, device='cuda'):
    training_data = FrimagenetDataset(spectrogram_folder, xception_folder)
    trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)

    hidden = model.init_hidden(batch_size)
    for h in hidden:
        h = h.to(device)    
    
    print_every = 20
    i = 0
    losses = []
    accs = []
    running_loss = 0.0
    running_acc = 0.0

    for epoch in range(epochs):
        for inp, labels in trainloader:  # renamed sequence to inp because inp is a batch of sequences
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            inp = inp.float().to(device)
            labels = labels.to(device)
            
            # Step 2. Run our forward pass.
            tag_scores, h = model(inp, hidden)

            # Step 3. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(tag_scores, labels)
            loss.backward()
            optimizer.step()

            running_acc += torch.mean((tag_scores.argmax(dim=1) == labels).float()).item()

            # print statistics
            running_loss += loss.item()
            if i % print_every == print_every-1:
                print('[%d, %5d] loss: %.3f - acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / print_every, running_acc * 100 / print_every))
                
                losses.append(running_loss / print_every)
                accs.append(running_acc * 100 / print_every)
                
                running_loss = 0.0
                running_acc = 0.0
            i += 1
    return losses, accs