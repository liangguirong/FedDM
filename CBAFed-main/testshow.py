import matplotlib.pyplot as plt
import torch
import os
import torchvision.utils as tvu
from torchvision import transforms
if __name__ == '__main__':
    images_train = torch.load(os.path.join("logged_files/CIFAR10", 'images_0.pt'))
    labels_train = torch.load(os.path.join("logged_files/CIFAR10", 'labels_0.pt'))
    im = images_train[0].permute(1, 2, 0)
    feat_image = transforms.ToPILImage()(im)
    feat_image.save('logged_files/image2.png')
    feat_tensor = images_train[0].permute(1, 2, 0)
    feat_tensor = (feat_tensor - feat_tensor.min()) / (feat_tensor.max() - feat_tensor.min()) * 32
    feat_tensor = feat_tensor.to(torch.uint8)
    feat_image = transforms.ToPILImage()(feat_tensor)
    feat_image.save('logged_files/image2.png')
    # plt.imshow(images_train[0].numpy().transpose(1, 2, 0))
    # plt.savefig('image.png')
    # image = im.transpose(1, 2, 0)
    # tvu.save_image(
    #     image, os.path.join("logged_files", f"{0}_{1}.png")
    # )
    # image.save('image1.png')