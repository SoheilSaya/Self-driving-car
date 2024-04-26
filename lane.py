import cv2
import numpy as np

def detect_edges(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    return edges

# Create a black mask with the same size as the image
def create_mask(image):
    return np.zeros_like(image)

# Read the input image
image = cv2.imread('road_image.jpg')

# Detect edges
edges = detect_edges(image)

# Get the height of the image
height = image.shape[0]

# Define the points of the triangle
pts = np.array([[200, height], [1100, height], [550, 250]], np.int32)

# Reshape the points to match the shape required by fillPoly
pts = pts.reshape((-1, 1, 2))

# Create a mask
mask = create_mask(edges)

# Fill the polygon with white color (255)
cv2.fillPoly(mask, [pts], 255)

# Apply the mask on the edges image
masked_edges = cv2.bitwise_and(edges, mask)

# Detect lines using Hough Line Transform with adjusted parameters
lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=100, maxLineGap=50)

#lines = cv2.HoughLinesP(masked_edges, 2, np.pi/180,100,np.array([]), minLineLength=40, maxLineGap=5)

# Draw detected lines on the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

# Display the image with detected lines
cv2.imshow('Detected Lines', image)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
