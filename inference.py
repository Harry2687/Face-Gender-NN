import modules.models as models
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

model_dir = 'trained_models/resnet_128_male/resnet_128_epoch_4.pt'
data_dir = 'data/Validation/'
classes = ('Female', 'Male')
imsize = 128

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.set_default_device(device)

loader = transforms.Compose([
    transforms.Resize([imsize, imsize]),
    transforms.Grayscale(1),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

external_test_dataset = datasets.ImageFolder(
    root=data_dir,
    transform=loader
)

n_images = min(len(external_test_dataset), 1000)

external_test_loader = DataLoader(
    external_test_dataset,
    batch_size=n_images,
    shuffle=True,
    generator=torch.Generator(device=device)
)

dataiter = iter(external_test_loader)
images, labels = next(dataiter)

test_cnn = models.resnetModel_128()
test_cnn.load_state_dict(torch.load(model_dir))

test_cnn.eval()
with torch.no_grad():
    cnn_output = test_cnn.forward(images.to(device))

y_pred = torch.max(cnn_output, 1)[1]

for i in range(n_images):
    print([img[0] for img in external_test_dataset.imgs][i])
    print(f'Prediction: {classes[y_pred[i]]}')
    print(f'{classes[0]} weight: {cnn_output[i][0]}')
    print(f'{classes[1]} weight: {cnn_output[i][1]}')