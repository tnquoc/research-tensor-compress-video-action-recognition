import os
import time
import warnings

import torch
import json
import numpy as np

from helpers import count_parameters
from utils import get_loaders
from model import CompressByTuckerVideoRecognizer

warnings.filterwarnings("ignore")


def main(params, device):
    train_loader, val_loader, test_loader = get_loaders(params, device)

    # input_shape = (164, 240, 320, 3)
    input_shape = (164, 120, 160, 3)
    tucker_ranks = params['tucker_ranks']

    model = CompressByTuckerVideoRecognizer(input_shape=input_shape, tucker_ranks=tucker_ranks, n_classes=24).to(device)
    count_parameters(model)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    if params['pre_trained']:
        checkpoint = torch.load(params['pre_trained'], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    num_epochs = params['epochs']
    train_loss = []
    val_loss = []
    print('Start training...')
    print('Epochs:', num_epochs)
    print('Iterations per training epoch:', len(train_loader))
    print('Iterations per validation epoch:', len(val_loader))
    for epoch in range(num_epochs):
        print(f'####### Epoch [{epoch + 1}/{num_epochs}] #######')
        t0 = time.time()
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

            if (i + 1) % 10 == 0:
                print('[Training] Step [{}/{}], Loss: {:.4f}'.format(i + 1, len(train_loader), loss.item()))

        validation_loss = 0
        model.eval()
        with torch.no_grad():
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

                if (i + 1) % 10 == 0:
                    print('[Validation] Step [{}/{}], ValLoss: {:.4f}'.format(i + 1, len(val_loader), loss.item()))

        training_loss = training_loss / len(train_loader)
        validation_loss = validation_loss / len(val_loader)
        train_loss.append(training_loss)
        val_loss.append(validation_loss)
        print("Epoch[{}/{}] - Loss: {:.4f} - ValLoss: {:.4f}, ETA: {:.0f}s"
              .format(epoch + 1, num_epochs, training_loss, validation_loss, time.time() - t0))

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(os.getcwd(), f'training/checkpoint_epoch{epoch}.pth'))

    torch.save(torch.from_numpy(np.array(train_loss)), os.path.join(os.getcwd(), 'training/history/train_loss.pt'))
    torch.save(torch.from_numpy(np.array(val_loss)), os.path.join(os.getcwd(), 'training/history/val_loss.pt'))

    print('Done')


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device:', torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu')
    if device != 'cpu':
        torch.multiprocessing.set_start_method('spawn')

    params = json.load(open('config.json', 'r'))

    main(params, device)

