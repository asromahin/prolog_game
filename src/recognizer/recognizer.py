import io

from google.cloud import vision
from google.cloud.vision_v1 import types

import cv2
from PIL import Image, ImageDraw

from google.oauth2 import service_account
import easyocr


class PageRecognizer(object):

    def __init__(self, recognizer, preprocess=None, ocr_model=None):
        self.recognizer = recognizer
        self.ocr_model = ocr_model
        self.preprocess = preprocess

    def get_min_rect(self, points):
        f_y = list(map(int, [v[0] for v in points]))
        f_x = list(map(int, [v[1] for v in points]))
        return [min(f_y), max(f_y)], [min(f_x), max(f_x)]

    def recognize(self, image):

        if self.preprocess:
            image = self.preprocess(image)

        bounds = self.recognizer.recognize_text(image)

        if self.ocr_model:
            for i, bound in enumerate(bounds):
                bound = list(bound)

                points = bound[0]
                y_v, x_v = self.get_min_rect(points)

                crop = image[x_v[0]:x_v[1], y_v[0]:y_v[1]]

                # plt.figure()
                # plt.imshow(crop)
                # print(bound)
                bound[1] = self.ocr_model.recognize(crop)
                bounds[i] = bound

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

    def __init__(self, client):

        self.client = client

    def recognize(self, image) -> str:

        cv2.imwrite('/content/temp.png', image)

        with io.open('/content/temp.png', 'rb') as image_file:
            content = image_file.read()
            imagesr = types.Image(content=content)

        response = self.client.text_detection(image=imagesr)

        if response.error.message:
            print('Error')

        texts = response.text_annotations

        if len(texts) == 0:
            return ""

        return texts[0].description.replace("\n", "")


class Recognizer(object):

    def __init__(self, reader, kwargs):
        self.kwargs = kwargs
        self.reader = reader

    def recognize_text(self, image):
        return self.reader.readtext(image, **self.kwargs)

    def detect(self, image):
        return self.reader.detect(image, **self.kwargs)


class RecognizePipeline:
    def __init__(self, credentials_path='/content/energy-meters-e2b1175c4c64.json'):
        api_key = service_account.Credentials.from_service_account_file(credentials_path)
        self.client = vision.ImageAnnotatorClient(credentials=api_key)
        self.reader = easyocr.Reader(['ru', 'ru'])
        self.ocr_model = OcrModel(self.client)
        self.rec = Recognizer(self.reader, {'paragraph': False})


    def image_preprocess(self, image):
        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
        return image

    def recognize(self, image):
      image = self.image_preprocess(image)
      bounds = PageRecognizer(recognizer=self.rec, ocr_model=self.ocr_model, preprocess=self.image_preprocess).recognize(image)
      return bounds