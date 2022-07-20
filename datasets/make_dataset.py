from distutils import filelist
import os
import numpy as np
from pathlib import Path
import shutil
def remove_file(src,dst):
    filelist = os.listdir(src)
    for file in filelist:
        src_file = os.path.join(src,file)
        dst_file = os.path.join(dst,file)
        shutil.move(src_file,dst_file)
root_path = "datasets/test_BSD_2ms16ms"
ch_path = "RGB"
for root, dirnames, filenames in os.walk(root_path):
    # print(root)
    # print(dirnames)
    # print(filenames)
    if ch_path == Path(root).name:
        print(root)
        print(Path(root).parent)
        remove_file(root,Path(root).parent)
        os.rmdir(root)

    

    

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


