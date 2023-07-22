import torchvision.transforms as transforms


def build_transforms(crop_size=224, horizontal_flip=False, vertical_flip=False):
    transform_list = [transforms.Resize((crop_size, crop_size))]

    if horizontal_flip:
        transform_list += [transforms.RandomHorizontalFlip()]
    if vertical_flip:
        transform_list += [transforms.RandomVerticalFlip()]

    transform_list += [transforms.ToTensor()]

    return {
        "train": transforms.Compose(transform_list),
        "val": transforms.Compose(
            [transforms.Resize((crop_size, crop_size)), transforms.ToTensor()]
        ),
    }
