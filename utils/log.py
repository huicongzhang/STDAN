import logging
logger = logging.getLogger()
logger.setLevel('INFO')
BASIC_FORMAT = "%(levelname)s:%(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
chlr.setLevel('INFO')
logger.addHandler(chlr)
def info(mes):
    logger.info(mes)
def error(mes):
    logger.error(mes)
def warning(mes):
    logger.warning(mes)
def critical(mes):
    logger.critical(mes)
def setFileHandler(dir,mode):
    fhlr = logging.FileHandler(dir,mode=mode)
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
""" class Log(object):
    def __init__(self, path,mode='w'):
        self.logger = logging.getLogger()
        self.logger.setLevel('INFO')
        BASIC_FORMAT = "%(levelname)s:%(message)s"
        DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
        formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
        chlr = logging.StreamHandler()
        chlr.setFormatter(formatter)
        chlr.setLevel('INFO')
        fhlr = logging.FileHandler(path,mode=mode)
        fhlr.setFormatter(formatter)
        self.logger.addHandler(fhlr)
        self.logger.addHandler(chlr)
    def info(self,mes):
        self.logger.info(mes)
    def error(self,mes):
        self.logger.error(mes)
    def warning(self,mes):
        self.logger.warning(mes)
    def critical(self,mes):
        self.logger.critical(mes) """
