import cv2 as cv
import numpy as np


def skin_detection(input_image):
    print('skin_detection')

    R = input_image[:, :, 2]
    G = input_image[:, :, 1]
    B = input_image[:, :, 0]

    norm_image = np.stack([B, G, R], axis=2)

    R = input_image[:, :, 2].astype("float")
    G = input_image[:, :, 1].astype("float")
    B = input_image[:, :, 0].astype("float")

    RGB = R+G+B

    r = R / RGB
    g = G / RGB
    b = B / RGB

    # calculando limetes superiores e inferiores da cor da pele
    f1 = -1.376 * (r ** 2) + (1.0743 * r) + 0.2
    f2 = -0.776 * (r ** 2) + (0.5601 * r) + 0.18

    # calculando variavel w(hite) para indicar pixel branco
    w = (((r - 0.33)**2) + ((g - 0.33) ** 2)) > 0.001

    # identificando o que é pele
    skin_mask = np.where(
        ((g < f1) & (g > f2) & (w > 0.001)), 1, 0)

    # Converte a máscara de pele para uma máscara de bits.
    skin_mask = skin_mask.astype("uint8")

    # Aplica a máscara à imagem.
    masked_image = cv.bitwise_and(
        norm_image, norm_image, mask=skin_mask)

    return masked_image


def resize_image(original_img):
    print('resize_image')
    scale_percent = 300  # percent of original size
    width = int(original_img.shape[1] * scale_percent / 100)
    height = int(original_img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv.resize(original_img, dim, interpolation=cv.INTER_AREA)

    return resized


if __name__ == '__main__':
    try:
        print('Process Init')
        # carregar a imagem
        original_img = cv.imread("assets/pImg.png")
        # original_img = cv.imread("images/face-pic1.jpg")

        resized_img = resize_image(original_img)

        skin_detected_image = skin_detection(resized_img)

        cv.imshow("imagem com pele detectada", skin_detected_image)
        cv.waitKey(0)
    except Exception as e:
        print("Erro: ", e)
