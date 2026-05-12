
# IMPORT PART
import torchvision.transforms as T
import yaml

# Load configs
with open('../configs/augmentation.yaml', 'r') as file:
    config = yaml.safe_load(file)

CIFAR_MEAN = tuple(config['normalization']['cifar_mean'])
CIFAR_STD = tuple(config['normalization']['cifar_std'])
s = config['color_distortion']['s']

# SIMCLR VIEW GENERATOR CLASS 
class SimCLRViewGenerator(object):
    """
    A custom wrapper to generate multiple augmented views of a single image,
    following the SimCLR framework for contrastive learning.
    """
    def __init__(self, base_transform, n_views=2):
        # The core stochastic augmentation pipeline to be applied to each input.
        self.base_transform = base_transform
        # Number of augmented views (typically 2 for creating positive pairs in SimCLR).
        self.n_views = n_views

    def __call__(self, x):
        """
        Executes the transformation pipeline multiple times to produce different 
        stochastic realizations of the same input image 'x'.
        """
        # Returns a list of 'n_views' variations, facilitating the calculation of contrastive loss.
        return [self.base_transform(x) for i in range(self.n_views)]


