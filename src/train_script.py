from ultralytics import YOLO

# Carregar modelo YOLO
model = YOLO('../models/yolo11x.pt')

# Mensagem para iniciar o TensorBoard
print("Type 'tensorboard --logdir=runs/train --bind_all --port=6006 --reload_interval 1'")
print("in terminal to view real-time metrics")

# Treinamento
results = model.train(
    data='/home/phcastello/Documentos/Cursos/ReconhecimentoImagem/CursoYolo/Dataset_final/data.yaml',  # Aleterar para o caminho absoluto
    epochs=150,
    batch=64,
    imgsz=640,
    device='cpu', # MUDAR PARA "cuda" TODO
    patience=10,

    # Data augmentation tradicional
    augment=True,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10, translate=0.1, scale=0.5,
    shear=2, flipud=0.1, fliplr=0.5, mosaic=0.9,

    # Mixup ativado (CutMix será adicionado via Albumentations)
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
