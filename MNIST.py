import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchvision import datasets, transforms
import logging
import datetime
from torchsummary import summary


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define the model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Layer 1: Conv -> ReLU -> BatchNorm -> MaxPool -> 1x1 Conv
        self.Layer1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),       # (1, 28, 28) -> (8, 28, 28)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),                 # (8, 28, 28) -> (8, 14, 14)
            nn.Conv2d(8, 8, 1),
            nn.BatchNorm2d(8),                  # 1x1 Conv -> (8, 14, 14)
            nn.Dropout(0.15)
        )

        # Layer 2: Conv -> ReLU -> BatchNorm -> MaxPool -> 1x1 Conv
        self.Layer2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),      # (8, 14, 14) -> (16, 14, 14)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),                 # (16, 14, 14) -> (16, 7, 7)
            nn.Conv2d(16, 16, 1),
            nn.BatchNorm2d(16),                # 1x1 Conv -> (16, 7, 7)
            nn.Dropout(0.15)
        )

        # Layer 3: Conv -> ReLU -> BatchNorm
        self.Layer3 = nn.Sequential(
            nn.Conv2d(16, 32, 3),               # (16, 7, 7) -> (32, 5, 5)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.15)
        )

        # Layer 4: Conv -> ReLU -> BatchNorm
        self.Layer4 = nn.Sequential(
            nn.Conv2d(32, 32, 3),               # (32, 5, 5) -> (32, 3, 3)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.15)
        )

        # Layer 5: 1x1 Conv to reduce channels before FC layer
        self.Layer5 = nn.Sequential(
            nn.Conv2d(32, 32, 1),               # 1x1 Conv -> (16, 3, 3)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.15)
        )

        # Fully connected layer (flatten the feature map)
        self.fc = nn.Sequential(
            nn.Linear(32 * 3 * 3, 10)           # Flatten and output 10 classes
        )

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.Layer5(x)
        x = x.view(-1, 32 * 3 * 3)  # Flatten the output of Layer5 (16 channels, 3x3 spatial size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Set device
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

model = Net().to(device)
summary(model, input_size=(1, 28, 28))

# Data Loaders
batch_size = 64
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                       # Reduced rotation angle
                        transforms.RandomRotation((-2, 2), fill=0),
                        # Very minimal translation
                        transforms.RandomAffine(
                            degrees=0,
                            translate=(0.02, 0.02),
                            fill=0
                        ),
                        # Added random erasing with small patches
                        transforms.RandomErasing(
                            p=0.1,
                            scale=(0.01, 0.02),
                            ratio=(0.3, 3.3),
                            value=0
                        )
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs
)

# Training and Testing Functions
best_test_acc = 0.0
best_model_state = None


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        accuracy = 100. * correct / total
        pbar.set_description(f"Epoch {epoch} - Loss: {loss.item():.4f} - Acc: {accuracy:.2f}%")
    print(f"Training Accuracy: {accuracy:.2f}%")


def test(model, device, test_loader):
    global best_test_acc, best_model_state
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    #test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    #print(f"Validation Accuracy: {accuracy:.2f}%")

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

    if accuracy > best_test_acc:
        best_test_acc = accuracy
        best_model_state = model.state_dict()
        print(f"New best model found with accuracy: {best_test_acc:.2f}%")

    return accuracy


if __name__ == "__main__":
# Model Training and Saving
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(0, 18):  # Train for 17 epochs
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    # Save the best model
    if best_model_state is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'mnist_best_model_{timestamp}.pth'
        torch.save({
            'model_state_dict': best_model_state,
            'Test_accuracy': best_test_acc,
            'epoch': epoch  # Save the current epoch
        }, save_path)
        print(f"Best model saved to {save_path}")
