import cv2
import os
import numpy as np

# Cargar la imagen original con los recortes
image_name = "La vuelta al hogar - Mendilaharzu Graciano_sin_recortes.png"
image_path = os.path.join("", image_name)
image = cv2.imread(image_path)

# Crear una máscara para los huecos (áreas negras o transparentes)
# Aquí asumimos que los huecos tienen píxeles de valor 0 (negro)
mask = cv2.inRange(image, (0, 0, 0), (0, 0, 0))

# Aplicar el inpainting usando el método Telea o Navier-Stokes
inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# Guardar o mostrar la imagen rellena
cv2.imwrite("image_filled.png", inpainted_image)
cv2.imshow("Imagen Rellena", inpainted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
