from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

modelo_carregado = load_model("modelo_mnist.h5")

def prever_digito(caminho_imagem):
    from PIL import Image, ImageOps
    import cv2

    # 1. Abrir e converter para escala de cinza
    img = Image.open(caminho_imagem).convert('L')

    # 2. Inverter (fundo preto, número branco)
    img = ImageOps.invert(img)

    # 3. Converter para array e aplicar limiarização
    img = np.array(img)
    _, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # 4. Encontrar contorno do dígito
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    img_cortada = img[y:y+h, x:x+w]

    # 5. Redimensionar mantendo proporção
    img_redimensionada = cv2.resize(img_cortada, (20, 20), interpolation=cv2.INTER_AREA)

    # 6. Colocar em uma imagem 28x28 e centralizar
    img_final = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - 20) // 2
    y_offset = (28 - 20) // 2
    img_final[y_offset:y_offset+20, x_offset:x_offset+20] = img_redimensionada

    # 7. Normalizar e preparar para input do modelo
    img_final = img_final.astype('float32') / 255.0
    img_final = img_final.reshape(1, 28, 28, 1)

    # 8. Carregar modelo e prever
    modelo = load_model("modelo_mnist.h5")
    predicao = modelo.predict(img_final)
    numero = np.argmax(predicao)

    print(f"Predição do modelo: {numero}")


# Exemplo de uso da função de previsão

# Caminho da imagem a ser testada
caminho_imagem = 'C:\GitHub\Deep-Learning-Personal-Project\imagem_nove.png'  # Substitua pelo caminho da sua imagem

# Chama a função de previsão
prever_digito(caminho_imagem)