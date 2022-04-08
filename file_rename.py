import os

format_type = {'type_1':[], 'type_2':[]}
folders = [file for file in os.listdir() if '.' not in file]

for folder in folders:
    axial_files   = [file for file in os.listdir(folder) if (file.endswith('.jpg') and file.startswith('0'))]
    coronal_files = [file for file in os.listdir(folder) if (file.endswith('.jpg') and file.startswith('1'))]
    for file in axial_files:
        os.replace(folder+'/'+file, folder+'/axial/'+file)
    for file in coronal_files:
        os.replace(folder+'/'+file, folder+'/coronal/'+file)
