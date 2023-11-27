import hair_detection_module
import skin_detection_module
import cv2 as cv
from matplotlib import pyplot as plt

# Utilizando camerâ não inbutida no PC
# Enumera todas as câmeras disponíveis
# for i in range(0, 10):
#     cap = cv.VideoCapture(i, cv.CAP_DSHOW)
#     if cap.isOpened():
#         print(f"Câmera USB encontrada no índice {i}")
#         cap.release()

# Abre a câmera USB desejada (substitua o índice conforme necessário)
# cap = cv.VideoCapture(1, cv.CAP_DSHOW)

# Inicializa a captura de vídeo (0 para câmera padrão, pode variar dependendo da sua configuração)
cap = cv.VideoCapture(0)

# Verifica se a captura foi inicializada com sucesso
if not cap.isOpened():
    print("Não foi possível abrir a câmera.")
    exit()

# Loop de captura de vídeo
while True:
    # Captura um novo frame
    ret, frame = cap.read()

    # Verifica se a leitura do frame foi bem-sucedida
    if not ret:
        print("Falha ao capturar o frame.")
        break

    # result_frame = hair_detection_module.hair_detection(frame)
    result_frame = skin_detection_module.skin_detection(frame)

    # Exibe o frame em uma janela
    cv.imshow("Video", result_frame)

    # Verifica se a tecla 'q' foi pressionada para encerrar o loop
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Libera os recursos ao finalizar
cap.release()
cv.destroyAllWindows()
