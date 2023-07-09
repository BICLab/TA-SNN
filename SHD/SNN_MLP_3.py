import os
import sys

rootPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

from SHD.MLP import SNN_3, Config


class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


logPath = Config.configs().recordPath
if not os.path.exists(logPath):
    os.makedirs(logPath)
sys.stdout = Logger(logPath + os.sep + "log_SHD_MLP_SNN_3.txt")


def main():
    SNN_3.main()


if __name__ == '__main__':
    main()