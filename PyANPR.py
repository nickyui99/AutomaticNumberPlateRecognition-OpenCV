import cv2
from skimage.measure import shannon_entropy
import numpy as np

class PyANPR:
    """
   PyANPR is a class that gather requiring image preprocessing techniques inspired by JavaANPR

   Attributes:
       K_SQUARE_S: Square kernel 3X3
       K_SQUARE_M: Square kernel 5X5
       K_SQUARE_L: Square kernel 9X9
   """
    K_SQUARE_S = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    K_SQUARE_M = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    K_SQUARE_L = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    # Constructor
    # def __init__(self):

    # To show image
    def imshow(title, image, wait_key=False):
        cv2.imshow(title, image)
        if wait_key:
            cv2.waitKey(0)

    def rgb_to_grayscale(image):
        """
        Convert RGB to grayscale
            OpenCV default using weighted average Y←0.299⋅R+0.587⋅G+0.114⋅B

        Parameters:
            image (Mat): Input RGB image.

        Returns:
            return_type: Gray scale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def clahe(image, clip_limit=2.0, tile_grid_size=(10, 10)):
        """
        Apply Contrast Limited Adaptive Histogram Equalization

        Parameters:
            image (Mat): Input image.
            clip_limit (Int): This parameter sets the threshold for contrast limiting. The default value is 2.
            tile_grid_size (Int): This sets the number of tiles in the row and column. By default this is 8×8. It is used while the image is divided into tiles for applying CLAHE.

        Returns:
            return_type: image.
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    def create_light_img(gray, d=5, sigma_color=20, sigma_space=75):
        """
        Generate a binary image. It can be used to mask during the morphological process

        Parameters:
            gray (Mat): Input image.
            d (Int): Dimension k size for bilateral filter. The default value is 5.
            sigma_color (Int): Sigma color for bilateral filter. Lower value preserve more edges. The default value is 20.
            sigma_space (Int): Sigma space for bilateral filter. Lower value preserve more edges. The default value is 75.

        Returns:
            return_type: image.
        """
        light = cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space) #Small value of sigma color and dimension helps to preserve more edges
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return light

    def canny_edge(gray,  d=3, sigma_color=20, sigma_space=75, thres1=150, thres2=200):
        """
        Detect edges using canny edge detection

        Parameters:
            gray (Mat): Input image.
            d (Int): Dimension k size for bilateral filter. The default value is 5.
            sigma_color (Int): Sigma color for bilateral filter. Lower value preserve more edges. The default value is 20.
            sigma_space (Int): Sigma space for bilateral filter. Lower value preserve more edges. The default value is 75.
            thres1(int): Threshold 1 of canny edge detection. The default value is 150.
            thres2 (int): Threshold 2 of canny edge detection. The default value is 200.

        Returns:
            return_type: image.
        """
        blur = cv2.bilateralFilter(gray, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        edges = cv2.Canny(blur, threshold1=thres1, threshold2=thres2)
        return edges

    def crop_roi(image, x, y, w, h, m=5):
        """
        To crop region of interest in our image, a 15 margin of pixel is given by default

        Parameters:
            image (Mat): Input image.
            d (Int): The x coordinate of ROI.
            y (Int): The y coordinate of ROI.
            w (Int): The width of ROI.
            h (int): The height of ROI.
            m (int): Margin of ROI. The default value is 15.

        Returns:
            return_type: image.
        """

        # Add a 15-pixel margin, ensuring the number plates stay within the image dimensions
        margin = m
        start_y = max(0, y - margin)
        start_x = max(0, x - margin)
        end_y = min(image.shape[0], y + h + margin)
        end_x = min(image.shape[1], x + w + margin)

        roi = image[start_y:end_y, start_x:end_x]
        return roi, start_y, end_y, start_x, end_x

    def compute_stats(image):
        mean = image.mean()
        std_dev = image.std()
        entropy = shannon_entropy(image)
        return mean, std_dev, entropy

    def calculate_psnr(original, processed):
        # Ensure both images are of the same size
        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return float('inf')  # Perfect match
        max_pixel = 255.0  # Assuming 8-bit image
        psnr = 10 * np.log10((max_pixel ** 2) / mse)
        return psnr
