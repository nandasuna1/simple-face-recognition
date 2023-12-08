import utils as u
import numpy as np
import cv2 as cv

# nesse modulo usaremos o elemento de intensidade do HSI para detectar a cor do cabelo


def hair_detection(input_image):
    R, G, B = u.split_RGB(input_image)

    R = R.astype("float")
    G = G.astype("float")
    B = B.astype("float")

    H, I = u.get_HSI_elements(input_image)

    condition_1 = np.bitwise_and(I < 80, np.bitwise_or(B - G < 15, B - R < 15))
    condition_2 = np.bitwise_and(20 < H, H <= 40)

    hair_mask = np.where(np.bitwise_or(condition_1, condition_2), 1, 0)
    # Converte a máscara de pele para uma máscara de bits.
    hair_mask = hair_mask.astype("uint8")

    # Aplica a máscara à imagem.
    masked_image = cv.bitwise_and(
        input_image, input_image, mask=hair_mask)

    gray_img = cv.cvtColor(masked_image, cv.COLOR_BGR2GRAY)

    return gray_img


def hair_quantization(hair_mask, square_size=5, threshold=12):
    # Inicializa a imagem quantizada de cabelo com zeros
    quantized_hair = np.zeros_like(hair_mask, dtype=np.uint8)

    # Loop pelos pixels da máscara de cabelo
    for y in range(0, hair_mask.shape[0], square_size):
        for x in range(0, hair_mask.shape[1], square_size):
            # Obtém o bloco 5x5 da máscara de cabelo
            block = hair_mask[y:y+square_size, x:x+square_size]

            # Conta o número de pixels brancos (valor 1) no bloco
            black_pixel_count = np.sum(block == 0)

            # Verifica se é um bloco de cor de cabelo ou não
            is_hair_block = black_pixel_count > threshold

            # Preenche o bloco correspondente na imagem quantizada de cabelo
            quantized_hair[y:y+square_size, x:x +
                           square_size] = is_hair_block.astype("uint8") * 255

    return quantized_hair
