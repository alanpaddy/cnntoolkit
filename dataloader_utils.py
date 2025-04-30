from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_set = datasets.FakeData(transform=transform)
    val_set = datasets.FakeData(transform=transform)
    return DataLoader(train_set, batch_size=batch_size), DataLoader(val_set, batch_size=batch_size)
