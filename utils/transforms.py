from torchvision import transforms
from torchvision.transforms import functional as TF
import random
import torch


def transform(data, mask):
    # Probability of applying flip transformations
    h_flip_prob = random.random()
    v_flip_prob = random.random()

    for sat in data:
        image, date = data[sat]

        # Transform to tensor
        image = torch.tensor(image)

        # Random crop
        # if random.random() > 0.5:
        #     original_shape = image.shape[-1]
        #     size = random.randint(original_shape*3//4, original_shape)
        #     i, j, h, w = transforms.RandomCrop.get_params(
        #         image, output_size=(size, size))
        #     image = TF.crop(image, i, j, h, w)
        #     mask = TF.crop(mask, i, j, h, w)
        #
        #     # # Resize
        #     resize = transforms.Resize(size=(original_shape, original_shape))
        #     image = resize(image)
        #     mask = resize(mask)

        # Random horizontal flipping
        if h_flip_prob > 0.5:
            image = TF.hflip(image)

        # Random vertical flipping
        if v_flip_prob > 0.5:
            image = TF.vflip(image)

        # Random Brightness
        if random.random() > 0.5:
            brightness_factor = random.uniform(0.5, 1.5)
            image = image * brightness_factor

        # Gaussian Blur
        if random.random() > 0.5:
            image = TF.gaussian_blur(image, [3, 3])

        # Random rotate
        # if random.random() > 0.5:
        #     angle = random.randint(-180, 180)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(image, angle)
        data[sat] = (image, date)

    mask = torch.tensor(mask)
    # Random horizontal flipping
    if h_flip_prob > 0.5:
        mask = TF.hflip(mask)
    # Random vertical flipping
    if v_flip_prob > 0.5:
        mask = TF.vflip(mask)
    return data, mask


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}