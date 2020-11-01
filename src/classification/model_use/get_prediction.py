import torchvision.transforms as transforms
import torch


class ClassificationModel:
    def __init__(self, model_path='src/classification/model_pt/mobilenet_v2_0.9873.pt'):
        self.model = torch.load(model_path).to('cpu')

    def transform_image(self, image):
        my_transforms = transforms.Compose([transforms.Resize(512, 512),
                                            transforms.ToTensor(),
                                            ])
        return my_transforms(image).unsqueeze(0)

    def get_category(self, image):
        transformed_image = self.transform_image(image=image)
        outputs = self.model.forward(transformed_image)

        category = torch.argmax(outputs).item()

        predicted_idx = int(category)
        return predicted_idx

    def predict(self,  image):
        image_with_tags = {0: 'БТИ', 1: 'ЗУ', 2: 'Разр. на ввод', 3: 'Разр. на стр-во', 4: 'Свид. АГР'}
        idx = self.get_category(image=image)
        tag_image = image_with_tags[idx]
        return tag_image

