from src.models import FPN
from src.config import FPN18Conf

if __name__ == '__main__':
    conf = FPN18Conf(num_classes=1000)
    model = FPN(conf)
    model.init_weights()

    print()

