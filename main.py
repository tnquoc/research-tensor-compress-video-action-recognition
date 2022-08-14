import os
import time

import torch
import json
import numpy as np

from utils import get_loaders
from model import CompressByTuckerVideoRecognizer


def main(params):
    t0 = time.time()
    train_loader, val_loader, test_loader = get_loaders(params)
    print(time.time() - t0)

    input_shape = (164, 240, 320, 3)
    tucker_ranks = params['tucker_ranks']
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(device)
    model = CompressByTuckerVideoRecognizer(input_shape=input_shape, tucker_ranks=tucker_ranks, n_classes=24).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    num_epochs = params['epochs']
    train_loss = []
    val_loss = []
    for epoch in range(num_epochs):
        training_loss = 0
        model.train()
        for i, samples in enumerate(train_loader):
            # Get data
            cores, factor_matrices_1, factor_matrices_2, factor_matrices_3, factor_matrices_4, labels = samples

            # clear gradient
            optimizer.zero_grad()

            # Forward pass
            outputs = model((cores.to(device),
                             factor_matrices_1.to(device),
                             factor_matrices_2.to(device),
                             factor_matrices_3.to(device),
                             factor_matrices_4.to(device)))

            # calculate loss
            loss = criterion(outputs.to(device), labels.to(device))

            # Backprop and optimize
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            if (i + 1) % 1 == 0:
                print("[Training] Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item()))

        validation_loss = 0
        model.eval()
        for i, samples in enumerate(val_loader):
            # Get data
            cores, factor_matrices_1, factor_matrices_2, factor_matrices_3, factor_matrices_4, labels = samples

            # Forward pass
            outputs = model((cores.to(device),
                             factor_matrices_1.to(device),
                             factor_matrices_2.to(device),
                             factor_matrices_3.to(device),
                             factor_matrices_4.to(device)))

            # calculate loss
            loss = criterion(outputs.to(device), labels.to(device))

            validation_loss += loss.item()

            if (i + 1) % 1 == 0:
                print("[Validation] Epoch[{}/{}], Step [{}/{}], ValLoss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, len(val_loader), loss.item()))

        training_loss = training_loss / len(train_loader)
        validation_loss = validation_loss / len(val_loader)
        train_loss.append(training_loss)
        val_loss.append(validation_loss)
        print("Epoch[{}/{}], Loss: {:.4f}, ValLoss: {:.4f}"
              .format(epoch + 1, num_epochs, training_loss, validation_loss))

        torch.save(model.state_dict(), os.path.join(os.getcwd(), f'training/model_weights_epoch{epoch}.pth'))

    torch.save(torch.from_numpy(np.array(train_loss)), 'train_loss.pt')
    torch.save(torch.from_numpy(np.array(val_loss)), 'val_loss.pt')


if __name__ == '__main__':
    params = json.load(open('config.json', 'r'))

    main(params)

