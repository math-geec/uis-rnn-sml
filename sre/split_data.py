import os, sys, shutil
import numpy as np

# Split the Fisher dataset into train set and test set randomly
# with ratio of 9:1

datadir, savedir = sys.argv[1:3]
train_path = os.path.join(savedir, "train_set")
test_path = os.path.join(savedir, "test_set")

if os.path.exists(train_path) or os.path.exists(test_path):
    print("Folders already created!")
    sys.exit()

speakers = [f for f in os.listdir(datadir) if os.path.isfile(os.path.join(datadir, f))]

os.makedirs(train_path)
os.makedirs(test_path)

np.random.shuffle(speakers)
num = len(speakers) // 10

for spk in speakers[:num]:
    old_path = os.path.join(datadir, spk)
    new_path = os.path.join(test_path, spk)
    shutil.copy(old_path, new_path)

for spk in speakers[num:]:
    old_path = os.path.join(datadir, spk)
    new_path = os.path.join(train_path, spk)
    shutil.copy(old_path, new_path)
