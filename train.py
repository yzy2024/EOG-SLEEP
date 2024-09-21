import torch
import matplotlib.pyplot as plt


def train_SE(model, device, train_loader, criterion, optimizer, epochs):
    model.to(device)
    # train_loader.to(device)
    model.train()
    acc_list = []
    for epoch in range(epochs):
        correct = 0
        total = 0
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 150 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{epochs}], Validation Accuracy: {accuracy:.2f}%')
        acc_list.append(accuracy)
    plt.plot(acc_list, label='acc', alpha=0.5)
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('aac')
    plt.legend()
    plt.show()
