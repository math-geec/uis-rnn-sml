import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# Analyse the distribution of utterance number in dataset

data_path = sys.argv[1]

utterance_number = [np.load(os.path.join(data_path, f), mmap_mode='r').shape[0] for f in os.listdir(data_path)]

bins = [0, 5, 10, 20, 50, 100, 500]
hist, bin_edges = np.histogram(utterance_number, bins)
fig, ax = plt.subplots()
# Plot the histogram heights against integers on the x axis
ax.bar(range(len(hist)), hist, width=1)

# Set the ticks to the middle of the bars
ax.set_xticks([0.5 + i for i, j in enumerate(hist)])

# Set the xticklabels to a string that tells us what the bin edges were
ax.set_xticklabels(['{} - {}'.format(bins[i], bins[i + 1]) for i, j in enumerate(hist)])

# plt.hist(utterance_number, bins=[0,5,10,20,50,100,500])
# plt.ylabel('Number of speakers')
# plt.xlabel('Number of utterance')
plt.savefig('Distribution.png')
