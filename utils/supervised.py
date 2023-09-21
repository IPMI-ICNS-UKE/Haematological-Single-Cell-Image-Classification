import random
from torchvision.transforms import transforms
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        blur_transforms = transforms.GaussianBlur(
            kernel_size=11, sigma=random.uniform(self.radius_min, self.radius_max)
        )
        return blur_transforms(img)

def map_classes(paths,labels,class_names_train):
    paths_keep = []
    labels_keep = []
    labels_not_keep = []
    for path,label in zip(paths,labels):
        if label in class_names_train or label == 'MYC':
            paths_keep.append(path)
            labels_keep.append(label)
        else:
            labels_not_keep.append(label)
    print(f'Kept classes {set(labels_keep)}')
    print(f'Dont kept {set(labels_not_keep)}')


    return paths_keep, labels_keep