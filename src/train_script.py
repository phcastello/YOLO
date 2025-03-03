from ultralytics import YOLO
import os
import torch

import torch

def find_batch_target(model, dataset, device, start_batch=8, batch_target=128):
    """Encontra o maior batch possível sem causar erro de memória."""
    batch = start_batch
    while batch <= batch_target:
        try:
            print(f"Testando batch size: {batch}")
            model.train(data=dataset, epochs=1, batch=batch, imgsz=640, device=device)
            batch *= 2  # Dobra o batch para testar o próximo nível
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                batch //= 2  # Reduz o batch para o último valor válido
                print(f"Memória insuficiente! Usando batch size: {batch}")
                return batch
            else:
                raise e  # Se for outro erro, levanta a exceção normalmente
    return batch

if __name__ == '__main__':
    # Verificar se há GPU disponível e definir o dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Carregar modelo YOLO
    model = YOLO('../models/yolo11x.pt')

    # dataset
    dataset = os.path.abspath('../Dataset_final/data.yaml')

    # Definir o batch suportado pela GPU
    #batch_target = 256

    arquivoLeitura = 'batch-txt.txt'

    if os.path.exists(arquivoLeitura):
        with open(arquivoLeitura, "r", encoding="utf-8") as file:
            batch_txt = file.read().strip()
            if batch_txt:
                batch_supported = batch_txt
            else:
                os.remove(arquivoLeitura)
    else:
        batch_supported = find_batch_target(model, dataset, device, start_batch=16, batch_target=256)

    with open(arquivoLeitura, "w", encoding="utf-8") as file:
        file.write(batch_supported)

    print(f'valor de batch usado: {batch_supported}')

    #accumulate = max(1, batch_target // batch_supported)
    # model.overrides.update({
    #     'accumulate': accumulate
    # })

    # Mensagem para iniciar o TensorBoard
    print("Type 'tensorboard --logdir=runs/train --bind_all --port=6006 --reload_interval 1'")
    print("in terminal to view real-time metrics")

    # Treinamento
    results = model.train(
        data=dataset,  # Aleterar para o caminho absoluto
        epochs=150,
        batch=batch_supported,
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
