import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Define the CNN architecture
class SkinCancerCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SkinCancerCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(in_features=16*16*32, out_features=128)  # Update in_features
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x, genetics=0,age='empty',sex='empty',location='empty'):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16*16*32)  # Update the size here
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = nn.functional.softmax(x)
        if (age != 'empty' and sex!='empty' and location!='empty'):
            lr=lr_model.predict_proba([[age,sex,location]])
            if genetics == 1:
                lr[0][0]-=0.10
                lr[0][1]+=0.10
            if lr[0][1] > lr[0][0]:
                x[0][0]-=0.10
                x[0][1]+=0.10
        else:
            if genetics == 1:
                x[0][0]-=0.05
                x[0][1]+=0.05
        return x

# Usage


# Data preprocessing and loading
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.ImageFolder(root='/kaggle/input/melanoma-skin-cancer-dataset-of-10000-images/melanoma_cancer_dataset/train', transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Initialize the model, loss function, and optimizer
model = SkinCancerCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

