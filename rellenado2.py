import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# Importar el modelo DeepFill
# Asumimos que has descargado el modelo y que está disponible en tu directorio
model = torch.load('deepfillv2_model.pth')  # Cargar el modelo preentrenado
model.eval()  # Modo de evaluación

# Ruta de la imagen con huecos (imagen con áreas negras o blancas para inpainting)
image_name = "imagen_con_huecos.png"
image_path = "ruta/a/tu/imagen/" + image_name

# Leer la imagen
image = Image.open(image_path).convert('RGB')

# Preprocesamiento
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

input_image = transform(image).unsqueeze(0)  # Añadir batch dimension
input_image = input_image.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Crear una máscara binaria para las áreas que deben ser rellenadas (asumimos que los huecos son negros)
image_np = np.array(image)
mask = np.zeros_like(image_np[:, :, 0])  # Crear una máscara negra
mask[image_np[:, :, 0] == 0] = 1  # Los píxeles negros se consideran huecos (puedes ajustar esto según la imagen)

# Convertir la máscara a tensor
mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # Hacerlo compatible con el modelo

# Inpainting usando el modelo DeepFill
with torch.no_grad():
    output_image = model(input_image, mask_tensor)  # Obtener la imagen rellena

# Convertir la imagen resultante a formato adecuado para mostrar
output_image = output_image.squeeze(0).cpu().numpy()
output_image = np.transpose(output_image, (1, 2, 0))  # Convertir a formato HWC
output_image = (output_image * 0.5 + 0.5) * 255  # Desnormalizar
output_image = np.clip(output_image, 0, 255).astype(np.uint8)

# Mostrar la imagen rellena
cv2.imshow('Imagen Rellena', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Guardar la imagen resultante
cv2.imwrite('imagen_rellena.png', output_image)
print("Imagen rellena guardada.")
