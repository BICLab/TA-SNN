import os
import sys
rootPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(rootPath)[0]
sys.path.append(rootPath)

from DVSGestures.CNN_10clips import TA_SNN_LIAF, Config



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
sys.stdout = Logger(logPath + os.sep + "log_DVS_Gesture_10clips_TA_SNN_LIAF.txt")


def main():
    TA_SNN_LIAF.main()


if __name__ == '__main__':
    main()