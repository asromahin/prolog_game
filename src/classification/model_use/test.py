from src.classification.model_use.get_prediction import ClassificationModel
import cv2

CM = ClassificationModel()
TEST_IMAGE_PATH = 'test_images/test.jpg'
im = cv2.imread(TEST_IMAGE_PATH)
print(CM.get_prediction(im))