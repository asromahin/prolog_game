from src.classification.model_use.get_prediction import ClassificationModel
from src.recognizer.recognizer import RecognizePipeline
from src.ner.ner import DeepPavlovPipeline
from src.key_value_recognizer.recognizer import KeyRecognize
import numpy as np


def extract_all_text(bounds):
  res_text = ''
  for b in bounds:
    res_text = ' '.join([res_text, b[1]])
  return res_text


class ResultPipeline(dict):
    def __init__(self):
        super(ResultPipeline, self).__init__()


class Pipeline:
    def __init__(
            self,
            cls_model_path=None,
            credential_path=None,
            seg_model_path=None,
    ):
        self.classification = ClassificationModel(cls_model_path)
        self.recognize = RecognizePipeline(credential_path)
        self.ner = DeepPavlovPipeline()
        self.key_rec = KeyRecognize(seg_model_path=seg_model_path)

    def predict(self, image, query='тип документа'):
        res = ResultPipeline()
        print(np.array(image).shape)
        res['type'], res['titul'] = self.classification.predict(image)
        res['bounds'] = self.recognize.predict(np.array(image))
        res['fields'] = self.key_rec.predict(res['bounds'], image, 0)
        res['full_text'] = extract_all_text(res['bounds'])
        res['ner_result'] = self.ner.predict(res['full_text'], query)
        return res