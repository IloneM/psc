from os import listdir#,rename
from os.path import isfile,isdir,join
from shutil import copyfile

inpath = 'mountpoint'
outpath = 'simple-wdir'

onlydirs = [f for f in listdir(inpath) if isdir(join(inpath, f))]

for d in onlydirs:
    for f in listdir(join(inpath, d, 'ISOL', 'NO')):
        copyfile(join(inpath, d, 'ISOL', 'NO', f), join(outpath, f))
