import torchvision.transforms as transforms
import torch


class ClassificationModel:
    def __init__(self, model_path='src/classification/model_pt/mobilenet_v2_0.9873.pt'):
        self.model = torch.load(model_path).to('cpu')

    def transform_image(self, image):
        my_transforms = transforms.Compose([transforms.Resize(512),
                                            transforms.ToTensor(),
                                            ])
        return my_transforms(image).unsqueeze(0)

    def get_category(self, image):
        transformed_image = self.transform_image(image=image)
        #outputs, out_titul = self.model.forward(transformed_image)
        outputs = self.model.forward(transformed_image)
        category = torch.argmax(outputs).item()
        #category_titul = torch.argmax(out_titul).item()

        predicted_idx = int(category)
        #predicted_titul_idx = int(category_titul)
        return predicted_idx#, predicted_titul_idx

    def predict(self,  image):
        image_with_tags = {
            0: 'Технический паспорт',
            1: 'Договор аренды земного участка',
            2: 'Разрешение на ввод Объекта капитального строительства',
            3: 'Разрешение на строительство',
            4: 'Свидетельство об утверждении архитектурно-градостроительного решения'
        }
        titul_with_tags = {0: 'no_titul', 1: 'titul'}
        #idx, titul_idx = self.get_category(image=image)
        idx = self.get_category(image=image)
        tag_image = image_with_tags[idx]
        #tag_titul_image = titul_with_tags[titul_idx]
        return tag_image#, tag_titul_image

