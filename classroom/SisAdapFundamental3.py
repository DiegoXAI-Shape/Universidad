import os
import torch
from torch import nn
import pandas as pd
from PIL import Image
from torch.nn import functional as f
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModeloV2(nn.Module):
    def __init__(self):
        super(ModeloV2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels= 32, out_channels=64, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))
        self.maxpool = nn.MaxPool2d((2, 2))
        self.dropoutC = nn.Dropout2d(p=0.2)
        self.dropoutL = nn.Dropout(p=0.45)

        self.avg = nn.AdaptiveAvgPool2d((1, 1)) # batch, 128, 1, 1

        self.dense1 = nn.Linear(in_features=128, out_features=64)  
        self.dense2 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        #Bloque convolcuional 1
        x = f.relu(self.conv1(x)) # 126
        x = f.leaky_relu(self.maxpool(self.conv2(x))) # 124 -> Pool = 62
        x = self.dropoutC(x)

        #Bloque convolucional 2
        x = f.relu(self.conv3(x)) # 60
        x = f.leaky_relu(self.maxpool(self.conv4(x))) # 58 / 2 = 29
        x = self.dropoutC(x)

        #Bloque convolucional 3
        x = f.relu(self.conv5(x)) #27
        x = f.leaky_relu(self.maxpool(self.conv6(x))) # 25 / 2 = 12
        x = self.dropoutC(x)

        x = self.avg(x)
        x = torch.flatten(x, 1)

        x = f.relu(self.dense1(x))
        x = self.dropoutL(x)
        x = self.dense2(x)

        return x

class CustomDS2(Dataset):
    def __init__(self, dataframe, images_dir, transform=None, target_transform=None):
        self.labels = dataframe
        self.directory = images_dir
        self.transform_data = transform
        self.target_transform = target_transform
        self.mapping = {"Cat": 0, "Dog": 1}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_df = self.labels.iloc[idx, 1]
        img_path = os.path.join(self.directory, label_df, self.labels.iloc[idx, 0]) 
        filename = self.labels.iloc[idx, 0]

        img = Image.open(img_path).convert('RGB')
        img_label = self.mapping[label_df]

        if self.transform_data: 
            if isinstance(self.transform_data, dict):
                transformacion = self.transform_data.get(label_df, None)
                if transformacion:
                    img = transformacion(img)
            else:
                img = self.transform_data(img)
        if self.target_transform: 
            img_label = self.target_transform(img_label)
        return img, img_label, filename

def predict(model, dataloader):
    model.eval() # Ponemos el modelo en modo evaluación
    model.to(device)
    
    # Inicializamos variables por si el dataloader estuviera vacío (seguridad)
    confianza = None
    predict_label = None

    with torch.no_grad():
        for image, label, filename in dataloader:
            image = image.to(device)
            # label = label.to(device) # No la necesitas para predecir

            predicts = model(image)
            prob = torch.nn.functional.softmax(predicts, 1)
            
            # CORRECCIÓN AQUÍ: Agregamos dim=1
            confianza, predict_label = torch.max(prob, 1)
            
            # Como batch_size=1, podemos retornar inmediatamente el primer resultado
            # Usamos .item() para devolver números de Python, no tensores
            return confianza.item(), predict_label.item()
            
    return 0.0, -1 # Retorno por defecto si falla algo

def main():
    modelo = ModeloV2()
    # Asegúrate que la ruta sea correcta (usaste r'' así que está bien)
    modelo.load_state_dict(torch.load(r'C:\Users\PC\Desktop\Universidad\Code\SisAdapt\classroom\Models\ModeloPrueba2025-11-19.pth'))
    
    # Mueve el modelo a GPU si es necesario antes de predecir (aunque predict lo hace)
    modelo.to(device) 

    data = {
        "filename":['1.jpg'],
        "label":['Cat']
    }
    df = pd.DataFrame(data)
    
    transformador = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    Ds = CustomDS2(dataframe = df, images_dir=r'C:\Users\PC\Desktop\Road_to_Hackathon\PetImages', transform=transformador)
    dl = DataLoader(dataset=Ds, batch_size=1, shuffle=False)

    confianza, predict_label = predict(model=modelo, dataloader=dl)
    
    print(f"Confianza: {confianza:.4f}, Clase Predicha: {predict_label}")

    if predict_label == 0:
        print("Dispensando comida de gato 🐱")
    else:
        print("Dispensando comida de perro 🐶")

if __name__ == '__main__':
    main()