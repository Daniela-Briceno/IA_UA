#Librerias necesarias
import cv2  #type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import random
import os
from tqdm import tqdm # type: ignore
from pathlib import Path
from skimage.feature import hog # type: ignore
from sklearn.preprocessing import MinMaxScaler # type: ignore
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision import datasets, transforms # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.svm import SVC # type: ignore
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay # type: ignore
# Lee el Datasets: contiene imagenes de diferentes tipos de basura, organizadas en carpetas por categoria/clase.

# Lee la carpeta y sus sub-carpetas, elige imagen al azar
ruta_dataset = "garbage_classification"
base_path = Path(ruta_dataset)
clases = os.listdir(base_path)
clase_elegida = random.choice(clases)
imagenes = os.listdir(os.path.join(base_path, clase_elegida))
imagen_elegida = random.choice(imagenes)
ruta_imagen = os.path.join(base_path, clase_elegida, imagen_elegida)

# Preprocesamiento con OpenCV ----------------------------------------------------------------------------------------

# Leer imagen en color y en escala de grises
imagen_color = cv2.imread(ruta_imagen)
imagen_rgb = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2RGB)
imagen_gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)

# Filtros convolucionales: Suavizado (filtro Gaussiano)
suavizada = cv2.GaussianBlur(imagen_gris, (5, 5), 0)

# Deteccion de bordes (Canny)
bordes = cv2.Canny(suavizada, 50, 150)

# Umbralizacion
_, umbral = cv2.threshold(imagen_gris, 127, 255, cv2.THRESH_BINARY)

# Operaciones morfologicas
kernel = np.ones((5, 5), np.uint8)
erosionada = cv2.erode(umbral, kernel, iterations=1)
dilatada = cv2.dilate(umbral, kernel, iterations=1)
apertura = cv2.morphologyEx(umbral, cv2.MORPH_OPEN, kernel)
cierre = cv2.morphologyEx(umbral, cv2.MORPH_CLOSE, kernel)

imagenes = [imagen_rgb, suavizada, bordes, umbral, erosionada, dilatada, apertura, cierre]
titulos = ["Imagen RGB", "Suavizada(Gauss)", "Bordes(Canny)", "Umbral Binario", "Erosion", "Dilatacion", "Apertura", "Cierre"]

# Muestra la imagen modificada
plt.figure(figsize=(10, 5))
for i in range(len(imagenes)):
    plt.subplot(2, 4, i + 1)
    cmap = 'gray' if len(imagenes[i].shape) == 2 else None
    plt.imshow(imagenes[i], cmap=cmap)
    plt.title(titulos[i])
    plt.axis("off")
plt.suptitle("Preprocesamiento de Imagen con OpenCV", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Se recorren todas las imagenes y se extraen las caracteristicas numericas relevantes, y se almacenan para su uso posterior
datos = []
for clase in tqdm(os.listdir(base_path)):
    carpeta = base_path / clase
    if carpeta.is_dir():
        for img_path in carpeta.glob("*.jpg"):
            try:
                # Leer imagen en escala de grises
                img = cv2.imread(str(img_path), 0)
                img = cv2.resize(img, (128, 128))
                # HOG (Histogram of Oriented Gradients), para capturar bordes y texturas.
                hog_features = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=False, feature_vector=True)
                # Momentos de Hu, que representan la forma de la imagen, y son invariantes a escala, rotacion y traslacion.
                momentos = cv2.moments(img)
                hu_moments = cv2.HuMoments(momentos).flatten()
                # Calcula un histograma de intensidad de 64 bins y lo normaliza.
                histograma = cv2.calcHist([img], [0], None, [64], [0, 256]).flatten()
                histograma = histograma / np.sum(histograma)
                # Unir todos los vectores
                caracteristicas = np.concatenate([hog_features, hu_moments, histograma])
                datos.append({
                    "ruta": str(img_path),
                    "clase": clase,
                    "vector": caracteristicas
                })
            except Exception as e:
                print(f"Error procesando {img_path}: {e}")


# Toma la lista datos y la transforma en un DataFrame
# Crear DataFrame final
df_vectores = pd.DataFrame(datos)

# DataFrame de vectores
df_vectores = pd.DataFrame(datos)
X = pd.DataFrame(df_vectores["vector"].to_list())

# Normalizacion
scaler = MinMaxScaler()
X_normalizado = scaler.fit_transform(X)

# Union con etiquetas
df_final = pd.concat([df_vectores[["ruta", "clase"]].reset_index(drop=True), pd.DataFrame(X_normalizado, columns=[f"f{i}" for i in range(X.shape[1])])], axis=1)

# Guardar CSV final con caracteristicas normalizadas y etiquetas
csv_path = "caracteristicas_normalizadas.csv"
df_final.to_csv(csv_path, index=False)
print(f" Archivo CSV creado exitosamente: {csv_path}")
print(df_final.head())

# Implementacion de Modelos ------------------------------------------------------------------------------------------
print('Entrenamiento de modelo SVC en proceso...')

# SVC: Modelo Supervizado para clasificar datos en distintas categorias, buscando el mejor limite (frontera) que los separe.

# Separar caracteristicas y etiquetas
X = df_final.drop(columns=["ruta", "clase"])
y = df_final["clase"]

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Modelo SVC
modelo_svm = SVC(kernel='rbf', C=10, gamma='scale')
modelo_svm.fit(X_train, y_train)

# Prediccion
y_pred = modelo_svm.predict(X_test)
print('Entrenamiento de terminado')

# CNN (Red neuronal convolucional): Modelo deep learning especializada en procesamiento de imagenes
# Una CNN trabaja directamente con las imagenes reales a diferencia de SVC
print('Entrenamiento de CNN en proceso...')

#Define la ruta al dataset
transformaciones = transforms.Compose([
    transforms.Grayscale(),              # convierte RGB a escala de grises
    transforms.Resize((128, 128)),       # cambia el tamano de las imagenes
    transforms.ToTensor()                # convierte a tensor PyTorch y normaliza [0,1]
])

# Carga y transforma las imagenes
dataset = datasets.ImageFolder(root=ruta_dataset, transform=transformaciones)

# Division en entrenamiento(80%) y validacion(20%)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Crea los lotes para entrenamiento y validacion
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

#Modelo CNN
class CNN(nn.Module):
    def __init__(self, num_clases):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_clases)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 128 -> 64
        x = self.pool(F.relu(self.conv2(x)))  # 64 -> 32
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelo = CNN(num_clases=len(dataset.classes)).to(device)
criterio = nn.CrossEntropyLoss()
optimizador = optim.Adam(modelo.parameters(), lr=0.001)

# Bucle de entrenamiento
for epoch in range(2):  # Recorrer el conjunto de datos varias veces
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # Poner a cero los gradientes de los parámetros
        optimizador.zero_grad()
        # forward
        outputs = modelo(inputs)
        #calcula la perdida
        loss = criterio(outputs, labels)
        # Backpropagation
        loss.backward()
        optimizador.step()

        running_loss += loss.item()
        
    print(f"Epoca {epoch+1}, perdida: {running_loss/len(train_loader):.4f}")


print('Entrenamiento de terminado')

# Evaluacion y Comparacion -------------------------------------------------------------------------------------------

print(" Resumen del modelo SVC:")
print(classification_report(y_test, y_pred))
precision_svc = accuracy_score(y_test, y_pred)
print(f"Precision del modelo SVC: {precision_svc * 100:.2f}%")

modelo.eval()
correctos = 0
total = 0
y_truecnn = []
y_predcnn = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = modelo(inputs)
        _, predicciones = torch.max(outputs, 1)
        y_truecnn.extend(labels.cpu().numpy())
        y_predcnn.extend(predicciones.cpu().numpy())
        correctos += (predicciones == labels).sum().item()
        total += labels.size(0)

print(" Resumen del modelo CNN:")
print(classification_report(y_truecnn, y_predcnn, target_names=dataset.classes))
print(f" Precision en validacion: {100 * correctos / total:.2f}%")

# Matriz de confusion SVC
cm = confusion_matrix(y_test, y_pred, labels=modelo_svm.classes_)
disp = ConfusionMatrixDisplay(cm, display_labels=modelo_svm.classes_)
disp.plot(xticks_rotation=90)
plt.title("Matriz de Confusion - SVC")
plt.show()

# Matriz de confusion
cm = confusion_matrix(y_truecnn, y_predcnn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dataset.classes)
disp.plot(xticks_rotation=90)
plt.title("Matriz de Confusion - CNN")
plt.show()