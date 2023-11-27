import cv2 as cv
import numpy as np


def resize_image(original_img, percentual_size=100):
    print('resize_image')
    scale_percent = percentual_size  # percent of original size
    width = int(original_img.shape[1] * scale_percent / 100)
    height = int(original_img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv.resize(original_img, dim, interpolation=cv.INTER_AREA)

    return resized


def split_RGB(input_image):
    R = input_image[:, :, 2]
    G = input_image[:, :, 1]
    B = input_image[:, :, 0]

    return R, G, B


def get_HSI_elements(input_image):

    R, G, B = split_RGB(input_image)
    R = R.astype("float")
    G = G.astype("float")
    B = B.astype("float")

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

    I = (1/3) * (R+G+B)

    return H, I
