import cv2 as cv
import numpy as np
import utils as u


def skin_detection(input_image):
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

    # Converte a imagem para escala de cinza
    gray_image = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)

    # Aplica a máscara à imagem.
    masked_image = cv.bitwise_and(
        gray_image, gray_image, mask=skin_mask)

    return masked_image


def skin_quantization(skin_mask):
    # Cria uma imagem preta do mesmo tamanho da máscara de pele
    quantized_skin = np.zeros_like(skin_mask)

    # Tamanho do quadrado 5x5
    square_size = 5

    # Limiar para contar pixels não da cor da pele
    threshold = 12

    # Loop pelos pixels da máscara
    for y in range(0, skin_mask.shape[0], square_size):
        for x in range(0, skin_mask.shape[1], square_size):
            # Obtém o bloco 5x5 da máscara de pele
            block = skin_mask[y:y+square_size, x:x+square_size]

            # Conta o número de pixels não da cor da pele no bloco
            non_skin_count = np.sum(block != 0)

            # Define se o bloco é pele ou não pele com base no limiar
            is_skin_block = non_skin_count <= threshold

            # Preenche o bloco na imagem quantizada
            quantized_skin[y:y+square_size, x:x +
                           square_size] = is_skin_block.astype("uint8") * 255

    return quantized_skin
