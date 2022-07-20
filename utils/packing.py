import os, tarfile

def packing(tarfilename, dirname):    # tarfilename是压缩包名字，dirname是要打包的目录
    if os.path.isfile(dirname):
        with tarfile.open(tarfilename, 'w') as tar:
            tar.add(dirname)
    else:
        with tarfile.open(tarfilename, 'w') as tar:
            filepath = []
            for root, dirs, files in os.walk(dirname):
                for single_file in files:
                    # if single_file != tarfilename:
                    filepath.append(os.path.join(root, single_file))
            for file in filepath:
                if '.git' in file or "weights" in file or "exp_log" in file or '__pycache__' in file:
                    continue
                tar.add(file)