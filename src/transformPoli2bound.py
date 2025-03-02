import os

# Diretórios
labels_dir = "/home/phcastello/Documentos/Cursos/ReconhecimentoImagem/CursoYolo/Dataset_final/valid/labels"

for label_file in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, label_file)

    new_lines = []
    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        
        
        parts = line.strip().split()
        if not parts:
            print(f"Linha vazia encontrada: '{line.strip()}'")  # Depuração
            continue
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))

        if len(coords) == 4:  # Já é uma bounding box no formato correto
            new_lines.append(line)
        else:  # Convertendo polígono para bounding box
            x_points = coords[0::2]  # Pegando os valores pares (x)
            y_points = coords[1::2]  # Pegando os valores ímpares (y)

            # Calculando bounding box mínima
            x_min, x_max = min(x_points), max(x_points)
            y_min, y_max = min(y_points), max(y_points)

            # Convertendo para formato YOLO (x_center, y_center, width, height)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min

            new_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

    # Salvando novo arquivo de label com bounding boxes
    with open(label_path, "w") as f:
        f.writelines(new_lines)

print("Conversão de polígonos para bounding boxes concluída.")
