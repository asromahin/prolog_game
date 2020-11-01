import torch
import numpy as np
from src.docs.DocumentDescription import list_of_documents
from fuzzywuzzy import fuzz
import pandas as pd
import cv2


def get_min_rect(points):
    f_y = list(map(int, [v[0] for v in points]))
    f_x = list(map(int, [v[1] for v in points]))
    return [min(f_x), max(f_x)], [min(f_y), max(f_y)]


def get_center(points):

    y_v, x_v = get_min_rect(points)

    c_y = y_v[0] + (y_v[1] - y_v[0]) / 2
    c_x = x_v[0] + (x_v[1] - x_v[0]) / 2

    return np.array([c_x, c_y])


class KeyRecognize:
    def __init__(self, seg_model_path):
        self.model = torch.load(seg_model_path, map_location='cuda:0')

    def predict(self, bounds, input_image, doc_type):
        df = pd.DataFrame()

        for b in bounds:
            df = df.append({'bbox': b[0], 'text': b[1]}, ignore_index=True)

        image = np.array(input_image)

        cuda0 = torch.device('cuda:0')

        image = cv2.resize(image, (1024, 1024))

        image = image.astype(np.float32) / 255

        image = torch.tensor(image).to(device=cuda0).permute(2, 0, 1)

        image = torch.unsqueeze(image, 0)

        output = self.model(image).cpu().squeeze()

        output = output.permute(1, 2, 0)

        output = output.detach().numpy()

        image = np.array(input_image)

        image = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))

        IMAGE_SHAPE = image.shape[:2]

        class_arr = []

        for i, row in df.iterrows():

            y_v_box, x_v_box = get_min_rect(row['bbox'])

            values = np.zeros((output.shape[2]))

            for i in range(output.shape[2]):
                ch = output[:, :, i].copy()

                x_v = (np.array(x_v_box) / IMAGE_SHAPE[1]) * 1024

                y_v = (np.array(y_v_box) / IMAGE_SHAPE[0]) * 1024

                ch = ch[int(y_v[0]):int(y_v[1]), int(x_v[0]):int(x_v[1])]

                ch[ch > 0.5] = 1
                ch[ch <= 0.5] = 0

                values[i] = np.sum(ch) / (ch.shape[0] * ch.shape[1])

            ind = np.argmax(values)

            class_arr.append(ind)

        df['class'] = class_arr

        for l in list_of_documents:
            if l.rus_name == doc_type:
                document = l

        found_fields = dict()
        for f in document.fields[4:]:

            rus_name = f.rus_name

            max_v = -1

            for i, row in df.iterrows():

                dist = fuzz.ratio(row['text'].lower(), rus_name.lower())

                if dist > max_v and row['class'] == 1:
                    found_fields[f.rus_name] = row
                    max_v = dist

        for key, field_row in found_fields.items():

            points = field_row['bbox']

            goal_c = get_center(points)

            min_dist = 10000

            for i, row in df.iterrows():
                if row['class'] != 2:
                    continue
                c = get_center(row['bbox'])

                dist = np.sqrt(np.sum(np.power(c - goal_c, 2)))

                y_field_value = goal_c[1]
                y_value = c[1]

                if dist < min_dist and np.abs(y_field_value - y_value) < 10:
                    min_dist = dist
                    field_row['value'] = row['text']

            new_found = []
            for f in found_fields:
                if f.get('value') is not None:
                    new_found.append(f)

        return new_found