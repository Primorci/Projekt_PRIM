import numpy as np
import cv2
import os


def transform(image, prob_crop=0.5, prob_flip=0.5, prob_contrast=0.5, prob_scaledown=0.5,
              prob_rotation=0.5, prob_gaussian_blur=0.5, prob_noise=0.5, prob_color_jitter=0.5,
              prob_motion_blur=0.5):
    # Naredimo kopijo slike, da ne spremenimo originalne slike
    transformed_image = np.copy(image)
    augmented_data = {}  # Initialize an empty dictionary for augmented data
    
    # Verjetnost obreza
    if np.random.rand() < prob_crop:
        transformed_image = crop_image(transformed_image)
    
    # Verjetnost horizontalnega zrcaljenja
    if np.random.rand() < prob_flip:
        transformed_image = flip_image(transformed_image)
    
    # Verjetnost spremembe kontrasta
    if np.random.rand() < prob_contrast:
        transformed_image = adjust_contrast(transformed_image)
        
    # Verjetnost zmanjšanja ločljivosti
    if np.random.rand() < prob_scaledown:
        transformed_image = apply_scaledown(transformed_image)
    
    # Verjetnost rotacije
    if np.random.rand() < prob_rotation:
        transformed_image = rotate_image(transformed_image)
    
    # Verjetnost Gaussian Blur
    if np.random.rand() < prob_gaussian_blur:
        transformed_image = apply_gaussian_blur(transformed_image)
    
    # Verjetnost dodajanja šuma
    if np.random.rand() < prob_noise:
        transformed_image = add_noise(transformed_image)
    
    # Verjetnost barvnega popravka
    if np.random.rand() < prob_color_jitter:
        transformed_image = adjust_color(transformed_image)
    
    # Verjetnost motion blur
    if np.random.rand() < prob_motion_blur:
        transformed_image = apply_motion_blur(transformed_image)
    
    # Add the transformed image to the dictionary with key 'image'
    augmented_data['image'] = transformed_image
    
    # Vrnemo dictionary z augmentiranim podatkom
    return augmented_data



def crop_image(image):
    return image[100:400, 100:400] 

def flip_image(image):
    return cv2.flip(image, 1)

def apply_scaledown(image):
    return cv2.resize(image, (200, 150), interpolation=cv2.INTER_AREA)

def adjust_contrast(image, alpha_range=(0.5, 1.5)):
    img_float = image.astype(np.float32) / 255.0
    factor = 1.5
    img_contrasted = cv2.multiply(img_float, np.array([factor]))
    img_contrasted = np.clip(img_contrasted, 0, 1)
    return (img_contrasted * 255).astype(np.uint8)

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def rotate_image(image):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = 45 
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  
    return cv2.warpAffine(image, M, (w, h))

def apply_gaussian_blur(image):
    kernel_size = (5, 5)
    sigma = 0
    return cv2.GaussianBlur(image, kernel_size, sigma)

def add_noise(image):
    mean = 0
    stddev = 25
    noise = np.random.normal(mean, stddev, image.shape).astype(np.uint8) 
    return cv2.add(image, noise) 

def adjust_color(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    brightness_factor = 1.5 
    hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2] * brightness_factor, 0, 255)

    saturation_factor = 1.5 
    hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_factor, 0, 255)
    
    return cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

def apply_motion_blur(image, kernel_size=15):
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    # konvolucija
    return cv2.filter2D(image, -1, kernel)


def save_images_to_folder(images, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"obdelna_slika_{i}.jpg")
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Naložimo slike iz mape
    folder_path = "slike"
    images = load_images_from_folder(folder_path)
    
    # Uporabimo funkcijo transform() na vsaki sliki
    transformed_data = [transform(image) for image in images]
    
    output_folder = "obdelane_slike"
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    
    for i, data in enumerate(transformed_data):
        # Get the augmented image from the dictionary
        augmented_image = data['image']
        
        # Write the augmented image to the output folder
        output_path = os.path.join(output_folder, f"obdelna_slika_{i}.jpg")
        cv2.imwrite(output_path, augmented_image)

