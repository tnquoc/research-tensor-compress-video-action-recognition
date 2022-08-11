import torch
import json

from utils import get_loaders
from model import CompressByTuckerVideoRecognizer


if __name__ == '__main__':
    params = json.load(open('config.json', 'r'))

    data_loader = get_loaders(params)

    input_shape = (164, 240, 320, 3)
    tucker_ranks = params['tucker_ranks']
    device = 'cpu'
    model = CompressByTuckerVideoRecognizer(input_shape=input_shape, tucker_ranks=tucker_ranks, n_classes=24)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    num_epochs = params['epochs']
    for epoch in range(num_epochs):
        for i, samples in enumerate(data_loader):
            # Get data
            cores, factor_matrices_1, factor_matrices_2, factor_matrices_3, factor_matrices_4, labels = samples

            optimizer.zero_grad()

            # Forward pass
            outputs = model((cores, factor_matrices_1, factor_matrices_2, factor_matrices_3, factor_matrices_4))
            print(labels, outputs)
            loss = criterion(outputs, labels)

            # Backprop and optimize
            loss.backward()
            optimizer.step()

            if (i + 1) % 50 == 0:
                print("Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}"
                      .format(epoch + 1, num_epochs, i + 1, len(data_loader), loss.item()))

        torch.save(model.state_dict(), f'training/model_weights_epoch{epoch}.pth')
