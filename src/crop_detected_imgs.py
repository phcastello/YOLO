from ultralytics import YOLO
import shutil
import time
import os
import cv2
import numpy as np

def getComplementaryColor(color):
    # Complementary color calculation, ensure values are within 0-255 range
    comp_color = tuple(np.clip(255 - np.array(color), 0, 255).astype(int))
    return comp_color

model = YOLO("yolo11x.pt")

inicio = time.time()
image_path = "/home/phcastello/Downloads/laranjeira-uov.jpg"
img = cv2.imread(image_path)
results = model(img)

# Obtendo as bounding boxes, confianças e classes
boxes = results[0].boxes.xyxy.cpu().numpy()
confidences = results[0].boxes.conf.cpu().numpy()
class_indices = results[0].boxes.cls.cpu().numpy().astype(int)
class_names = model.names
print(len(class_names))

threshold = 0.3

# Diretório de saída para imagens recortadas
output_dir = "cropped_imgs"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

imgWithBoxes = img.copy()

# Desenhar bounding boxes com a cor complementar
for i, (box, conf, class_idx) in enumerate(zip(boxes, confidences, class_indices)):
    if conf > threshold:
        x_min, y_min, x_max, y_max = map(int, box[:4])

        # Recortar e salvar a imagem detectada
        cropped_img = img[y_min:y_max, x_min:x_max]
        class_label = class_names[class_idx]
        cropped_filename = os.path.join(output_dir, f"cropped_{class_label}_{i}.jpg")
        cv2.imwrite(cropped_filename, cropped_img)
        
        # Calcular a média da cor da região ao redor da bounding box
        roi = img[y_min:y_max, x_min:x_max]
        avgColor = np.mean(roi, axis=(0, 1))  # Média por canal (BGR)
        
        color = getComplementaryColor(avgColor)
        
        # Garantir que a cor esteja no formato correto: uma tupla de inteiros dentro do intervalo [0, 255]
        color = tuple(map(int, np.clip(color, 0, 255)))  # Convertendo para inteiros e forçando o intervalo correto
        
        # Verificando se a cor é válida
        if isinstance(color, tuple) and len(color) == 3 and all(isinstance(c, int) and 0 <= c <= 255 for c in color):
            # Desenhar a bounding box com a cor complementar
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness=1)

            # Pegar o índice da classe
            font = cv2.FONT_HERSHEY_TRIPLEX
            font_scale = 0.5
            cv2.putText(img, str(i), (x_min, y_min - 5), font, font_scale, color, thickness=1)
        else:
            print(f"Erro: Cor inválida para a bounding box {i}. Cor calculada: {color}")

fim = time.time()
print(f'Tempo total de processamento: {fim-inicio:.2} segundos')

cv2.imshow("Detections", img)

while True:
    key = cv2.waitKey(1)
    if cv2.getWindowProperty("Detections", cv2.WND_PROP_VISIBLE) < 1:
        break

cv2.destroyAllWindows()
