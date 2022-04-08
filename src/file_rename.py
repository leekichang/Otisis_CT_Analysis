from config import *
import os

folders = os.listdir()

for folder in folders:
    axial_files   = [file for file in os.listdir(folder) if (file.endswith('.jpg') and '_' not in file)]
    coronal_files = [file for file in os.listdir(folder) if (file.endswith('.jpg') and '_' not in file)]
    break
print(axial_files)