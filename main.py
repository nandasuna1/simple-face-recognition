import skin_detection_module as skin_d
import hair_detection_module as hair_d
import utils as u
import cv2 as cv

if __name__ == '__main__':
    try:
        print('Process Init')
        # carregar a imagem
        original_img = cv.imread("assets/pImg.png")
        # original_img = cv.imread("images/face-pic1.jpg")

        resized_img = u.resize_image(original_img, 300)

        skin_detected_image = skin_d.skin_detection(resized_img)

        hair_detect_image = hair_d.hair_detection(resized_img)

        cv.imshow("imagem com pele detectada", skin_detected_image)
        cv.imshow("imagem com cabelo detectado", hair_detect_image)

        cv.waitKey()
    except Exception as e:
        print("Erro: ", e)
