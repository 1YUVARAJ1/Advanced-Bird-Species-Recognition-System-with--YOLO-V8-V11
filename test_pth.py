import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import os
from torch.utils.data import DataLoader

def test_model():
    data_dir = 'data/processed/CUB_200_2011_cropped'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if os.path.exists(os.path.join(data_dir, 'test')):
        test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model = models.efficientnet_b3(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 200)
        )
        model.load_state_dict(torch.load('src/models/weights/efficientnet_b3_best.pth', map_location=device))
        model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        conf_sum = 0
        print("Running eval on first batch...")
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                max_probs, preds = torch.max(probs, 1)
                
                correct += torch.sum(preds == labels).item()
                total += labels.size(0)
                conf_sum += torch.sum(max_probs).item()
                break # just first batch
                
        print(f"Batch Acc: {correct/total*100:.2f}%")
        print(f"Avg Conf: {conf_sum/total*100:.2f}%")
        
        for i in range(5):
            print(f"Label: {labels[i].item()}, Pred: {preds[i].item()}, Conf: {max_probs[i].item()*100:.2f}%")
    else:
        print("Test data not found.")

if __name__ == '__main__':
    test_model()
