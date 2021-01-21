from torchvision.datasets import mnist

class FashionMNIST(mnist):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

        Args:
            root (string): Root directory of dataset where ``FashionMNIST/processed/training.pt``
                and  ``FashionMNIST/processed/test.pt`` exist.
            train (bool, optional): If True, creates dataset from ``training.pt``,
                otherwise from ``test.pt``.
            download (bool, optional): If true, downloads the dataset from the internet and
                puts it in root directory. If dataset is already downloaded, it is not
                downloaded again.
            transform (callable, optional): A function/transform that  takes in an PIL image
                and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
                target and transforms it.
        """
    def train_data_setter(self):
        self.data = self.data
        return self.data
    
    resources = [
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
         "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
         "25c81989df183df01b3e8a0aad5dffbe"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
         "bef4ecab320f06d8554ea6380940ec79"),
        ("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
         "bb300cfdad3c16e7a12a480ee83cd310")
    ]
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
               'Shirt', 'Sneaker', 'Bag', 'Ankle boot']