from src.models import __dict__


def get_model(arch):
    return __dict__[arch]()
