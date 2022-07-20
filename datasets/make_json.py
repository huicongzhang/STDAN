import json
import os
import io
from distutils import filelist
import numpy as np
from pathlib import Path
import shutil
""" from utils.imgio_gen import readgen
import numpy as np """

def load_file_list(root_path, child_path = None, is_flatten=False):
    folder_paths = []
    filenames_pure = []
    filenames_structured = []
    num_files = 0
    for root, dirnames, filenames in os.walk(root_path):
        # print('root: ', root)
        # print('dirnames: ', dirnames)
        # print('filenames: ', filenames)
        if len(dirnames) != 0:
            if dirnames[0][0] == '@':
                del(dirnames[0])

        if len(dirnames) == 0:
            if root[0] == '.':
                continue
            # if child_path is not None and child_path != Path(root).name:
            if child_path is not None and (child_path in root) != True:
                continue
            folder_paths.append(root)
            filenames_pure = []
            for i in np.arange(len(filenames)):
                if filenames[i][0] != '.' and filenames[i] != 'Thumbs.db':
                    filenames_pure.append(os.path.join(root, filenames[i]))
            filenames_pure
            filenames_structured.append(np.array(sorted(filenames_pure), dtype='str'))
            num_files += len(filenames_pure)

    folder_paths = np.array(folder_paths)
    filenames_structured = np.array(filenames_structured, dtype=object)

    sort_idx = np.argsort(folder_paths)
    folder_paths = folder_paths[sort_idx]
    filenames_structured = filenames_structured[sort_idx]

    if is_flatten:
        if len(filenames_structured) > 1:
            filenames_structured = np.concatenate(filenames_structured).ravel()
        else:
            filenames_structured = filenames_structured.flatten()

    return folder_paths, filenames_structured, num_files
root_path = "/home/hczhang/datasets/BSD/BSD_3ms24ms"
child_path = "RGB"
folder_paths, filenames_structured, num_files = load_file_list(root_path,child_path)
filenames_structured_list = list(filenames_structured)
folder_paths_list = list(folder_paths)
print(len(filenames_structured_list))
print(len(filenames_structured_list[0]))
print(filenames_structured_list[100][0].split("/"))
print(folder_paths_list[100])
samples = []
file = io.open('BSD_3ms24msDeblur.json','w',encoding='utf-8')
for i in range(len(folder_paths_list)):
    folder_path = folder_paths_list[i]
    print(folder_path.split("/"))
    sample_sub = []
    if folder_path.split("/")[6] != "valid" and folder_path.split("/")[8] != "Blur":
        for j in range(len(filenames_structured_list[i])):
            sample_sub.append(filenames_structured_list[i][j].split("/")[-1].split(".")[0])
        l = {'name': folder_path.split("/")[7],'phase': folder_path.split("/")[6],'sample': sample_sub}
        samples.append(l)
js = json.dump(samples, file, sort_keys=False, indent=4)
# For VideoDeblur dataset
# file = io.open('VideoDeblur.json','w',encoding='utf-8')
# root = '/DeepVideoDeblurringDataset'
""" file = io.open('GoproDeblur.json','w',encoding='utf-8')
root = '/home/hczhang/datasets/GOPRO_Large'


samples = []
phase = ['train', 'test']
for ph in phase:
    names = sorted(os.listdir(os.path.join(root, ph)))
    for name in names:
        sample_list = sorted(os.listdir(os.path.join(root, ph, name, 'blur_gamma')))
        sample = [sample_list[i][:-4] for i in range(len(sample_list))]
        sample_sub = []
        for sam in sample:
            if not sam == ".DS_S":
                sample_sub.append(sam)
        l = {'name': name,'phase': ph,'sample': sample_sub}
        samples.append(l)

js = json.dump(samples, file, sort_keys=False, indent=4) """



