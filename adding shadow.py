import cv2
import numpy as np

FG_IMG_PATH = "fg.png"
BG_IMG_PATH = "bg.jpeg"
BLUR_AMOUNT = 32

def load_image(path, color_conversion=None):
    """Load an image and optionally convert its color."""
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if color_conversion:
        image = cv2.cvtColor(image, color_conversion)
    return image

def extract_alpha_channel(image):
    """Extract the alpha channel and the RGB channels from an image."""
    alpha_channel = image[:,:,3]
    rgb_channels = image[:,:,0:3]
    return alpha_channel, rgb_channels

def apply_blur_to_alpha(alpha, blur_amount):
    """Apply blur to the alpha channel."""
    return cv2.blur(alpha, (blur_amount, blur_amount))

def expand_and_normalize_alpha(alpha):
    """Expand alpha dimensions and normalize its values to the range [0,1]."""
    expanded_alpha = np.expand_dims(alpha, axis=2)
    repeated_alpha = np.repeat(expanded_alpha, 3, axis=2)
    normalized_alpha = repeated_alpha / 255
    return normalized_alpha

def create_shadow_on_bg(bg, alpha_blur):
    """Put shadow (based on blurred alpha) on top of the background."""
    black_canvas = np.zeros(bg.shape, dtype=np.uint8)
    shadowed_bg = (alpha_blur * black_canvas + (1 - alpha_blur) * bg).astype(np.uint8)
    return shadowed_bg

def composite_foreground_on_bg(fg, alpha, bg_with_shadow):
    """Put the foreground image on top of the background with shadow."""
    composited_image = (alpha * fg + (1 - alpha) * bg_with_shadow).astype(np.uint8)
    return composited_image

if __name__ == "__main__":
    # Load images and convert their color if necessary
    fg = load_image(FG_IMG_PATH, cv2.COLOR_BGRA2RGBA)
    bg = load_image(BG_IMG_PATH, cv2.COLOR_RGB2BGR)

    # Extract alpha and RGB channels from the foreground image
    alpha, fg_rgb = extract_alpha_channel(fg)

    # Blur the alpha channel to get the shadow
    alpha_blur = apply_blur_to_alpha(alpha, BLUR_AMOUNT)

    # Expand and normalize the blurred alpha for shadow calculation
    alpha_blur_normalized = expand_and_normalize_alpha(alpha_blur)

    # Create a version of the background with the shadow
    bg_with_shadow = create_shadow_on_bg(bg, alpha_blur_normalized)

    # Expand and normalize the original alpha for compositing foreground over background
    alpha_normalized = expand_and_normalize_alpha(alpha)

    # Composite the foreground on the shadowed background
    final_image = composite_foreground_on_bg(fg_rgb, alpha_normalized, bg_with_shadow)

    # Display the final image (optional)
    cv2.imshow("Final Image", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()