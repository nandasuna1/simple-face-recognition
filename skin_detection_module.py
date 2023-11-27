import cv2 as cv
import numpy as np
import utils as u


def skin_detection(input_image):
    print('skin_detection')

    R, G, B = u.split_RGB(input_image=input_image)

    R = R.astype("float")
    G = G.astype("float")
    B = B.astype("float")

    RGB = R+G+B
    r = R / RGB
    g = G / RGB

    # calculando limetes superiores e inferiores da cor da pele
    f1 = -1.376 * (r ** 2) + (1.0743 * r) + 0.2
    f2 = -0.776 * (r ** 2) + (0.5601 * r) + 0.18

    # calculando variavel w(hite) para indicar pixel branco
    w = (((r - 0.33)**2) + ((g - 0.33) ** 2)) > 0.001

    H, _ = u.get_HSI_elements(input_image)

    # identificando o que é pele
    skin_mask = np.where(
        ((g < f1) & (g > f2) & (w > 0.001) & (np.bitwise_or(H > 240, H <= 20))), 1, 0)

    # Converte a máscara de pele para uma máscara de bits.
    skin_mask = skin_mask.astype("uint8")

    # Aplica a máscara à imagem.
    masked_image = cv.bitwise_and(
        input_image, input_image, mask=skin_mask)

    return masked_image
