import cv2
import easyocr
import imutils
import numpy as np

from PyANPR import PyANPR

# VERSION 1 70/80 (0.875) accuracy
# 8 18 21 33 38 58 61 67 72 73

# VERSION 2 79/80 (0.9875) accuracy
# 8 39 45 48 56 67
# 39 45 48 56 67
# 67

#Initialization
anpr = PyANPR
custom_config = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' # Define alphanumeric character set
reader = easyocr.Reader(['en'], gpu=False)

# Iterate over the range of images
for i in range(1, 81):

    """
    ###Image Acquisition###
    """
    filename = f"vehicle_dataset//test_{i:03}.jpg"
    print(f"Processing {filename}...")

    try:
        image = cv2.imread(filename)
        if image is None:
            print(f"Image {filename} not found or unreadable.")
            continue

        """
        ###Image Preprocessing###
        """

        # Convert to grayscale
        gray = PyANPR.rgb_to_grayscale(image)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = PyANPR.clahe(gray)

        # Create a binary image for masking in morphological process
        light = PyANPR.create_light_img(clahe)
        # imshow("Light", light)

        """
        ###Detect Edges###
        """
        edges = PyANPR.canny_edge(clahe)
        thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        # thresh = cv2.dilate(thresh, kernel_medium, iterations=3)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        # thresh = cv2.erode(thresh, kernel_medium, iterations=5)
        # # imshow("Grad Erode/Dilate", thresh)
        #
        # thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        # thresh = cv2.erode(thresh, kernel_medium, iterations=2)
        # thresh = cv2.dilate(thresh, kernel_medium, iterations=3)
        #
        # thresh = cv2.bitwise_and(thresh, thresh, mask=light) #add back some light regions feature to prevent over erosion and dilation
        # thresh = cv2.dilate(thresh, kernel_small, iterations=1)
        # imshow("Final", thresh, waitKey=True)

        """
        ###Morphological Operations###
        """
        # TODO: VERSION 2
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, PyANPR.K_SQUARE_M, iterations=2)
        thresh = cv2.dilate(thresh, PyANPR.K_SQUARE_L, iterations=2)
        thresh = cv2.erode(thresh, PyANPR.K_SQUARE_L, iterations=2)

        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, PyANPR.K_SQUARE_M, iterations=2)
        thresh = cv2.erode(thresh, PyANPR.K_SQUARE_M, iterations=4)
        thresh = cv2.dilate(thresh, PyANPR.K_SQUARE_M, iterations=4)

        thresh = cv2.bitwise_and(thresh, thresh, mask=light)
        # PyANPR.imshow("Final", thresh, wait_key=True)

        """
        ###Contour Detection###
        """
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        """
        ###Contour Filtering###
        """
        found_plate = False
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            print(aspect_ratio, area)

            if 2 < aspect_ratio < 6 and 500 < area < 6500 :
                roi, start_y, end_y, start_x, end_x = PyANPR.crop_roi(gray, x, y, w, h)
                # PyANPR.imshow("roi", roi, wait_key=True)

                # OCR to detect text
                texts = sorted(reader.readtext(roi, detail=0, allowlist=custom_config))
                print(f"Detected text: {texts}")

                if len(texts) > 0:  # Validate detected text
                    if len(texts[len(texts) - 1]) > 3:
                        # Draw bounding box and label
                        result = image.copy()
                        result = cv2.rectangle(result, (start_x, start_y), (end_x, end_y), (0, 255, 0),
                                               2)  # add 10 pixel margin
                        result = cv2.putText(result, "Number Plate", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                             (0, 255, 0), 2)
                        found_plate = True
                        PyANPR.imshow("Result", result, wait_key=True)
                        break

        if not found_plate:
            print("No number plate detected.")
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Processing complete.")
