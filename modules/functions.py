import torchvision.transforms as transforms
import matplotlib.pyplot as plt

unloader = transforms.ToPILImage()

def imshow(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    plt.pause(0.001)