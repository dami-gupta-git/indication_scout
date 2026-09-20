import logging

logger = logging.getLogger(__name__)
DATA = {"alpha": 1, "beta": 2, "gamma": 3, "delta": 4, "epsilon": 5}


def add(a, b):
    total = a + b
    return total


class Thing:
    def __init__(self, name):
        self.name = name

    def shout(self):
        logger.info("hello %s", self.name)
