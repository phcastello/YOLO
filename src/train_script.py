from ultralytics import YOLO
import torch

# Verificar se há GPU disponível e definir o dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Carregar modelo YOLO
model = YOLO('../models/yolo11x.pt')

# Definir o batch suportado pela GPU
batch_supported = 32  # Ajuste esse valor se necessário (sempre uma potencia de 2)
batch_target = 64
accumulate = max(1, batch_target // batch_supported)  # Garante que seja pelo menos 1

# Mensagem para iniciar o TensorBoard
print("Type 'tensorboard --logdir=runs/train --bind_all --port=6006 --reload_interval 1'")
print("in terminal to view real-time metrics")

# Treinamento
results = model.train(
    data='/home/phcastello/Documentos/Cursos/ReconhecimentoImagem/CursoYolo/Dataset_final/data.yaml',  # Aleterar para o caminho absoluto
    epochs=150,
    batch=batch_supported,  # Usa o batch que a GPU suporta
    accumulate=accumulate,  # Acumulação de gradientes para usar um batch menor com efeito de um maior
    imgsz=640,
    device=device,
    patience=10,

    # Data augmentation tradicional
    augment=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10, translate=0.1, scale=0.5,
    shear=2, flipud=0.1, fliplr=0.5, mosaic=0.9,

    # Mixup ativado (CutMix será adicionado via Albumentations no data.yaml)
    mixup=0.2,

    # TensorBoard
    project='runs/train',
    name='yolo11_experiment',
)

# Validação
metrics = model.val()
print(metrics)

# Teste com uma imagem
testResult = model('../Dataset_final/test/images/000e4e7ed48c932d.jpg')
testResult.show()
testResult.save("output.jpg")

# Exibir informações do treinamento
print(f"Treinando com batch {batch_supported}, acumulação {accumulate}, equivalente a batch {batch_target}.")
