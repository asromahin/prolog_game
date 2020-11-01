import os
import sys
sys.path.append(os.getcwd())
os.system("nohup sh -c '" + sys.executable + " qat_train.py > log.txt" + "' &")