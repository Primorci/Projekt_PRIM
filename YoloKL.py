import cv2
import numpy as np
from skimage import feature, morphology
from skimage.morphology import dilation, erosion

def resize_image(image, target_size=640):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    final_image = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    top, left = (target_size - new_h) // 2, (target_size - new_w) // 2
    final_image[top:top + new_h, left:left + new_w] = resized_image
    return final_image

def intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    edges = feature.canny(clahe_img, sigma=1, low_threshold=10, high_threshold=50)
    edges = edges.astype(np.uint8) * 255  # Convert edges to uint8 type
    dilated = dilation(edges, morphology.square(3))
    eroded = erosion(dilated, morphology.square(3))
    return eroded

def find_contours(image):
    processed = preprocess_image(image)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def grabcut_segmentation(image, rect):
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    segmented_image = image * mask2[:, :, np.newaxis]
    return segmented_image