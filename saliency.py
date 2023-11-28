import cv2
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

# Load an image
image_path = r"C:\Users\daisy\UTD_Optimization_Research\Assets\cameracapture.png"  # Replace with the actual path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Initialize the Spectral Residual Saliency Detector
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

# Compute the saliency map
(success, saliencyMap) = saliency.computeSaliency(image)

# Rescale the saliency map to the range [0, 255]
saliencyMap = (saliencyMap * 255).astype("uint8")
cv2.imwrite("saliencyMap.png", saliencyMap)

# Display the original image and the saliency map
#plt.imshow(image)
#plt.show()
#plt.imshow(saliencyMap)