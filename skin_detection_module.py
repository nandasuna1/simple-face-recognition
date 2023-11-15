import cv2 as cv
import numpy as np
import math


def skin_detection(input_image):
    print('skin_detection')

    R = input_image[:, :, 2]
    G = input_image[:, :, 1]
    B = input_image[:, :, 0]

    norm_image = np.stack([B, G, R], axis=2)

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

    # para maior acuracia vamos adotar elementos HIS(Hue, Intensity and Saturation)
    divisor = 0.5*((R-G)+(R-B))
    dividendo = np.sqrt(((R-G)**2) + ((R-B)*(G-B)))

    # Evita divisão por zero
    divisor[divisor == 0] = 1e-10
    dividendo[dividendo < 1e-10] = 1e-10

    # Garante que o argumento de np.arccos esteja no intervalo [-1, 1]
    teta_arg = np.clip((divisor/dividendo), -1, 1)

    # Calcular arco cosseno
    teta = np.arccos(teta_arg)
    teta = np.degrees(teta)

    H = np.where(B <= G, teta,  360 - teta)

    # identificando o que é pele
    skin_mask = np.where(
        ((g < f1) & (g > f2) & (w > 0.001) & (np.bitwise_or(H > 240, H <= 20))), 1, 0)

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

        cv.waitKey()
    except Exception as e:
        print("Erro: ", e)
