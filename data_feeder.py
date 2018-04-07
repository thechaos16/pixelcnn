
from PIL import Image


class DataFeeder:
    def __init__(self, dataset, tr):
        self.dataset = dataset
        self.tr = tr

    def __getitem__(self, index):
        img, target = self.dataset['data'][index],\
                      self.dataset['label'][index]
        img = Image.fromarray(img.numpy(), mode='L')

        if self.tr is not None:
            img = self.tr(img)

        return img, target

    def __len__(self):
        return len(self.dataset['data'])
