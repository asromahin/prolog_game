import torch
import torchvision.transforms as transforms
from PIL import Image


def transform_image(image):
    my_transforms = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor(),
    ])
    return my_transforms(image)

DOCNAME2CLASS = { 0: 'БТИ',
                  1: 'ЗУ',
                  2: 'Разр. на ввод',
                  3: 'Разр. на стр-во',
                  4: 'Свид. АГР'}

model = torch.load('/home/tiptjuk/home/tiptjuk/pycharm_project/hack_lid_cif/model_train/experiments/mobilenet_v2/qat_models/29_0.9873.pt')

img = '/data/hackathon/pdf2image/БТИ/1 изм.pdf/0.jpg'
img = Image.open(img)
#img = np.array(img)

img = transform_image(img).to('cuda')
out = model.forward(img.unsqueeze(0))

idx = int(torch.argmax(out).item())
class_ = DOCNAME2CLASS[idx]
print(class_)