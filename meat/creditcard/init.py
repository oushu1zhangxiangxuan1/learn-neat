import os
import sys

PROC_PATH: str = os.path.abspath(os.path.realpath(__file__))
# LITTLEBOY_CV_HOME: str = os.path.abspath(
#     os.path.dirname(PROC_PATH)+os.path.sep+"../..")
DEV_PATH: str = os.path.abspath(
    os.path.dirname(PROC_PATH)+os.path.sep+"../../..")
sys.path.append(DEV_PATH)
# print(sys.path)

# print("LITTLEBOY_CV_HOME:", LITTLEBOY_CV_HOME)


def touch():
    pass
