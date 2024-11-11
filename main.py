import torch
import torchvision.transforms as T
import cv2
import numpy as np
import os
from PIL import Image

# Ruta de la imagen
image_name = "La vuelta al hogar - Mendilaharzu Graciano.jpg"
image_path = os.path.join("images", image_name)

# Definir el nombre base de los archivos recortados (sin extensión)
base_name = os.path.splitext(os.path.basename(image_path))[0]

# Verificar si la imagen existe
if not os.path.exists(image_path):
    print(f"Error: La imagen no existe en la ruta: {image_path}")
else:
    # Cargar la imagen original
    image = Image.open(image_path).convert("RGB")
    
    # Transformar la imagen para que pueda ser procesada por DeepLabV3
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Cargar el modelo DeepLabV3 preentrenado
    model = torch.hub.load("pytorch/vision:v0.10.0", "deeplabv3_resnet101", pretrained=True)
    model.eval()

    # Realizar predicción de segmentación
    with torch.no_grad():
        output = model(input_tensor)["out"][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

    # Filtrar la clase de "persona" (clase 15 en el modelo COCO)
    person_class = 15
    mask = output_predictions == person_class

    # Crear un contorno alrededor de cada detección de persona
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    coordinates = []

    # Dibujar los contornos y recortar las regiones detectadas
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        coordinates.append((x, y, w, h))

        # Dibujar un rectángulo rojo alrededor de cada detección
        cv2.rectangle(image_cv2, (x, y), (x + w, y + h), (0, 0, 255), 4)

        # Recortar y guardar la región detectada
        cropped_person = image_cv2[y:y+h, x:x+w]
        output_path = f"{base_name}_recorte_{i}.png"
        Image.fromarray(cropped_person).save(output_path)

    # Mostrar la imagen con los rectángulos dibujados
    cv2.imshow("Imagen con Detecciones de Personas", image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Guardar la imagen final con los rectángulos rojos
    output_image_path = f"{base_name}_con_rectangulos_rojos.png"
    cv2.imwrite(output_image_path, image_cv2)
    print(f"Imagen con detecciones guardada en: {output_image_path}")

    # Guardar las coordenadas en un archivo
    with open(f"{base_name}_coordenadas.txt", "w") as file:
        if coordinates:
            for coord in coordinates:
                file.write(f"{coord}\n")
            print(f"Coordenadas guardadas en: {base_name}_coordenadas.txt")
        else:
            print("No se detectaron personas, por lo tanto no se guardaron coordenadas.")
