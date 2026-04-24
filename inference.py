import torch
from model import SiameseUNet

def load_model():
    model = SiameseUNet(in_channels=5)
    model.eval()
    return model

def predict(model, img1, img2):
    img1 = torch.tensor(img1).unsqueeze(0).float()
    img2 = torch.tensor(img2).unsqueeze(0).float()

    with torch.no_grad():
        out = model(img1, img2)
        out = torch.sigmoid(out)
        

    return out.numpy()