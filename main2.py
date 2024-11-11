import torch
import cv2
import os
import numpy as np
from PIL import Image
import torchvision.transforms as T

# Cargar el modelo DeepLabV3 preentrenado
model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True)
model.eval()

# Ruta de la imagen
image_name = "La vuelta al hogar - Mendilaharzu Graciano.jpg"
image_path = os.path.join("images", image_name)
base_name = os.path.splitext(os.path.basename(image_path))[0]

# Verificar si la imagen existe
if not os.path.exists(image_path):
    print(f"Error: La imagen no existe en la ruta: {image_path}")
else:
    image = Image.open(image_path).convert("RGB")
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Convertir el output a una imagen binaria donde 15 = persona en COCO
    mask_person = (output_predictions == 15).astype(np.uint8) * 255
    
    # Convertir la imagen original a numpy para dibujar
    image_np = np.array(image)
    coordinates = []
    
    # Buscar contornos para cada área detectada
    contours, _ = cv2.findContours(mask_person, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        if w < 25 or h < 25:
            continue
        
        # Guardar las coordenadas del recorte
        coordinates.append((x, y, w, h))
        
        # Recortar y guardar cada persona
        cropped_object = image_np[y:y+h, x:x+w]
        output_path = f"{base_name}_recorte_{i}.png"
        Image.fromarray(cropped_object).save(output_path)
        
        # Rellenar el área en la imagen original con blanco
        image_np[y:y+h, x:x+w] = [255, 255, 255]

    # Guardar las coordenadas en un archivo
    with open(f"{base_name}_coordenadas.txt", "w") as file:
        for coord in coordinates:
            file.write(f"{coord}\n")
        print(f"Coordenadas guardadas en: {base_name}_coordenadas.txt")
    
    # Guardar la imagen final con los recortes rellenados en blanco
    output_image_path = f"{base_name}_sin_recortes.png"
    Image.fromarray(image_np).save(output_image_path)
    print(f"Imagen sin recortes guardada en: {output_image_path}")
