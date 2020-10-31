from recognize import Recognizer
from recognize import PageRecognizer
from recognize import OcrModel
import cv2
import matplotlib.pyplot as plt
import easyocr

def image_preprocess(image):

    return image


image = cv2.imread("../test_images/test.jpg")

image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))


reader = easyocr.Reader(['ru', 'ru'])

rec = Recognizer(reader, {'paragraph': False})

bounds = PageRecognizer(recognizer=rec, preprocess=image_preprocess).recognize(image)

image_with_bounds = PageRecognizer.draw_bounds(image, bounds)


plt.figure(figsize=(12, 12))
plt.imshow(image_with_bounds)
plt.show()