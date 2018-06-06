import os


def get_rgb_image():
    return os.path.join(os.path.dirname(__file__), 'rgb.jpg')


def get_grayscale_image():
    return os.path.join(os.path.dirname(__file__), 'grayscale.png')
