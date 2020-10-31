import PIL
from PIL import ImageDraw
from PIL import Image
import cv2
import numpy as np


class PageRecognizer(object):

    def __init__(self, recognizer, preprocess=None, ocr_model=None):
        self.recognizer = recognizer
        self.ocr_model = ocr_model
        self.preprocess = preprocess

    def get_min_rect(self, points):
        f_y = [v[0] for v in points]
        f_x = [v[1] for v in points]
        return [min(f_y), max(f_y)], [min(f_x), max(f_x)]

    def recognize(self, image):

        if self.preprocess:
            image = self.preprocess(image)
        
        bounds = self.recognizer.recognize_text(image)

        if self.ocr_model:
            for bound in bounds:
                points = bound[0]
                y_v, x_v = self.get_min_rect(points)
                crop = image[y_v[0]:y_v[1], x_v[0]:x_v[1]]
                bound[1] = self.ocr_model.recognize(crop)
        
        return bounds

    @staticmethod
    def draw_bounds(image, bounds, color='yellow', width=2):

        pil_image = Image.fromarray(image)

        draw = ImageDraw.Draw(pil_image)
        for bound in bounds:
            p0, p1, p2, p3 = bound[0]
            draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)

        return pil_image


class OcrModel(object):

    def __init__(self):
        pass

    def recoginze(self) -> str:
        pass


class Recognizer(object):

    def __init__(self, reader, kwargs):
        self.kwargs = kwargs
        self.reader = reader
        
    def recognize_text(self, image):
        return self.reader.readtext(image, **self.kwargs)


