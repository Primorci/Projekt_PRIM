import numpy as np
import cv2
import os
from scipy.ndimage import convolve


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



def crop_image(image, crop_height=200, crop_width=200):
    height, width = image.shape[:2]
    start_x = np.random.randint(0, width - crop_width)
    start_y = np.random.randint(0, height - crop_height)
    cropped_image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    return cropped_image

def flip_image(image):
    return cv2.flip(image, 1)

def apply_scaledown(image, scale_factor_range=(0.5, 0.8)):
    scale_factor = np.random.uniform(scale_factor_range[0], scale_factor_range[1])
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    return scaled_image

def adjust_contrast(image, alpha_range=(0.5, 1.5)):
    alpha = np.random.uniform(alpha_range[0], alpha_range[1])
    beta = np.random.uniform(0, 255)
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def rotate_image(image, max_angle=45):
    angle = np.random.uniform(-max_angle, max_angle)
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def apply_gaussian_blur(image, max_kernel_size=5):
    kernel_size = np.random.randint(1, max_kernel_size + 1)  # Ensure odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

def add_noise(image, noise_type='gaussian', mean=0, std=20, amount=0.01):
    noisy_image = np.copy(image).astype(float)  # Convert image to float
    
    if noise_type == 'gaussian':
        noise = np.random.normal(mean, std, noisy_image.shape)
        noisy_image += amount * noise
    elif noise_type == 'salt_and_pepper':
        mask = np.random.rand(*noisy_image.shape) < amount
        noisy_image[mask] = 0
        mask = np.random.rand(*noisy_image.shape) < amount
        noisy_image[mask] = 255
    elif noise_type == 'uniform':
        noise = np.random.uniform(-amount, amount, noisy_image.shape)
        noisy_image += noise
    
    return np.clip(noisy_image, 0, 255).astype(np.uint8)  # Convert back to uint8 before returning

def adjust_color(image, brightness_range=(-50, 50), contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_range=(-10, 10)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(float)
    
    # Adjust brightness
    brightness = np.random.uniform(brightness_range[0], brightness_range[1])
    hsv_image[:, :, 2] += brightness
    hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2], 0, 255)
    
    # Adjust contrast
    contrast = np.random.uniform(contrast_range[0], contrast_range[1])
    hsv_image[:, :, 2] *= contrast
    
    # Adjust saturation
    saturation = np.random.uniform(saturation_range[0], saturation_range[1])
    hsv_image[:, :, 1] *= saturation
    
    # Adjust hue
    hue = np.random.uniform(hue_range[0], hue_range[1])
    hsv_image[:, :, 0] += hue
    hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0], 0, 179)
    
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

def apply_motion_blur(image, kernel_size=15):
    # Generate motion blur kernel (horizontal direction)
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel /= kernel_size
    
    # Ensure the image has the correct shape
    if len(image.shape) == 3:  # Check if image is color (has channels)
        channels = image.shape[2]
        motion_blurred_image = np.zeros_like(image)
        for i in range(channels):
            motion_blurred_image[:, :, i] = convolve(image[:, :, i], kernel)
    else:  # Grayscale image
        motion_blurred_image = convolve(image, kernel)
    
    return motion_blurred_image.astype(np.uint8)



def save_images_to_folder(images, output_folder):
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    for i, image in enumerate(images):
        output_path = os.path.join(output_folder, f"augmented_image_{i}.jpg")
        cv2.imwrite(output_path, image)

if __name__ == "__main__":
    # Naložimo slike iz mape
    folder_path = "TestFrames"
    images = load_images_from_folder(folder_path)
    
    # Uporabimo funkcijo transform() na vsaki sliki
    transformed_data = [transform(image) for image in images]
    
    output_folder = "NewFrames"
    os.makedirs(output_folder, exist_ok=True)  # Create the output folder if it doesn't exist
    
    for i, data in enumerate(transformed_data):
        # Get the augmented image from the dictionary
        augmented_image = data['image']
        
        # Write the augmented image to the output folder
        output_path = os.path.join(output_folder, f"augmented_image_{i}.jpg")
        cv2.imwrite(output_path, augmented_image)

