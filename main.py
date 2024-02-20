import skin_detection_module as skin_d
import hair_detection_module as hair_d
import utils as u
import cv2 as cv
import numpy as np


def process_image(frame):
    print('Process Init')
    # carregar a imagem
    # original_img = cv.imread("assets/pImg.png")
    # original_img = cv.imread("images/face-pic1.jpg")

    # resized_img = u.resize_image(frame, 300)

    skin_detected_image = skin_d.skin_detection(frame)
    hair_detect_image = hair_d.hair_detection(frame)

    skin_quantization = skin_d.skin_quantization(skin_detected_image)
    hair_quantization = hair_d.hair_quantization(hair_detect_image)

    combined_mask = skin_quantization | hair_quantization

    skin_labeled = u.component_labeling_with_size_filter(
        combined_mask)
    skin_labeled = cv.bitwise_not(skin_labeled)

    # cv.imshow("Detecção de pele", skin_detected_image)
    # cv.imshow("Detecção de cabelo", hair_detect_image)
    # cv.imshow("Quantizaçãp de pele", skin_quantization)
    # cv.imshow("skin_labeled", skin_labeled)

    combined_image = np.zeros_like(frame)
    # Create a new mask with the same size and type as combined_image
    new_mask = np.zeros_like(combined_image)
    # Copy the values from skin_labeled to all channels of new_mask
    new_mask[:, :, :] = skin_labeled[:, :, None]
    # cv.imshow("imagem L", combined_image)
    combined_image = cv.bitwise_and(frame, new_mask)

    return combined_image


if __name__ == '__main__':
    try:
        # cv.imshow("imagem com pele detectado", new_mask)
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
            result_frame = process_image(frame)

            # Exibe o frame em uma janela
            cv.imshow("Video", result_frame)

            # Verifica se a tecla 'q' foi pressionada para encerrar o loop
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # Libera os recursos ao finalizar
        cap.release()
        cv.destroyAllWindows()
    except Exception as e:
        print("Erro: ", e)
