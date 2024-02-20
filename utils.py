import cv2 as cv
import numpy as np


def resize_image(original_img, percentual_size=100):
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


def component_labeling_with_size_filter(image,  min_area=20000):
    # Ensure image is grayscale
    if len(image.shape) > 2:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    # Check if input image has connected components
    if np.all(np.diff(gray_image) == 0):  # Check if all pixel differences are 0 (all same value)
        print("Warning: Input image has no connected components.")
        return gray_image  # Return the original image without labeling

    # Apply connected components labeling
    ret, labeled_image, stats, centroids = cv.connectedComponentsWithStats(
        gray_image, connectivity=8)
    # Calculate areas of components
    areas = stats[:, -1]  # Extract areas from stats

    # Filter components based on area
    filtered_labels = []
    filtered_stats = []
    stats_index = 0
    for label, area in zip(stats[:, 0], areas):
        if area >= min_area:  # Exclude background (label 0)
            filtered_labels.append(labeled_image[label])
            filtered_stats.append(stats[stats_index])
        stats_index += 1

    # Combine filtered components into a single mask
    combined_mask = np.zeros_like(labeled_image)
    for label in filtered_labels:
        combined_mask[labeled_image == label] = 255

    # Apply the mask to the original image
    filtered_image = image.copy()
    # filtered_image[combined_mask == 0] = 0

    filtered_image = filtered_image.astype(np.uint8)
    print("combined mask", filtered_labels)
    # Draw bounding boxes around filtered components
    for label, stat in zip(filtered_labels, stats):
        area = stat[-1]  # Extract areas from stats
        if area >= min_area:
            x, y, w, h, a = stat
            pt1 = (x, y)
            pt2 = (x + w, y + h)
            cv.rectangle(filtered_image, pt1, pt2, (0, 0, 0), 5)

    # Return the combined image
    return filtered_image


def calculate_component_features(labeled_image):
    features = []

    for label in np.unique(labeled_image):
        if label == 0:
            continue  # Pular rótulo de fundo

        # Encontrar pixels pertencentes ao componente
        component_mask = np.uint8(labeled_image == label)

        # Calcular área
        area = np.sum(component_mask)

        # Calcular centro de gravidade (COG)
        rows, cols = np.where(component_mask)
        cog = (np.mean(rows), np.mean(cols))

        # Encontrar coordenadas extremas
        min_row, max_row, min_col, max_col = np.min(
            rows), np.max(rows), np.min(cols), np.max(cols)

        # Armazenar características do componente
        component_features = {
            "label": label,
            "area": area,
            "cog": cog,
            "min_row": min_row,
            "max_row": max_row,
            "min_col": min_col,
            "max_col": max_col,
        }

        features.append(component_features)

    return features


def apply_size_filter(labeled_image, skin_component_features, size_threshold):
    # Encontrar o rótulo do maior componente de pele
    largest_skin_label = max(skin_component_features,
                             key=lambda x: x["area"])["label"]

    # Criar máscara do maior componente de pele
    largest_skin_mask = np.uint8(labeled_image == largest_skin_label)

    # Verificar se a área do maior componente de pele atende ao limite
    if skin_component_features[largest_skin_label]["area"] > size_threshold:
        # Aplicar Size Filter
        filtered_image = cv.bitwise_and(
            labeled_image, labeled_image, mask=largest_skin_mask)
        return filtered_image
    else:
        return labeled_image


# for label in np.unique(labeled_image):
#     if label == 0:
#         continue  # Pule o rótulo do fundo
#     label_mask = np.uint8(labeled_image == label) * 255
#     cv.imshow(f"Label {label}", label_mask)
