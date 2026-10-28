import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = "brain_mri_Y1.jpg"
image = cv2.imread(filename)

if image is None:
    print("Error: Could not read the image.")
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    ret, segmented = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, hierarchy = cv2.findContours(
        segmented,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = image.copy()
    cv2.drawContours(
        output,
        contours,
        -1,
        (255, 0, 0),
        2
    )
    print("Image loaded and segmented:", filename)

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image)
plt.title("Original MRI")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(blur, cmap="gray")
plt.title("Gaussian Blur")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(segmented, cmap="gray")
plt.title("Otsu Segmentation")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(output)
plt.title("Detected Contours")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(segmented, cmap="gray")
plt.title("Final Segmented Image")
plt.axis("off")

plt.tight_layout()
plt.show()

print("----- MRI Image Features ------")
print("Image Size:", image.shape)
print("Otsu Threshold:", ret)
print("Number of Contours:", len(contours))

if len(contours) > 0:
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    x, y, w, h = cv2.boundingRect(largest_contour)

    print("Largest Region Area:", area)
    print("Largest Region Perimeter:", perimeter)
    print("Bounding Box:")
    print("X:", x)
    print("Y:", y)
    print("Width:", w)
    print("Height:", h)
