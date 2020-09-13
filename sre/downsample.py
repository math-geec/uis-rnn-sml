import os
import subprocess
# import librosa
import numpy as np

# librosa==0.7.2
# numba==0.48.0

# # path to load original audio data
# ori_path = '/uclprojects/SpeakerDiarisation/ori_data'
# # path to save resampled audio data
# resample_path = '/uclprojects/SpeakerDiarisation/resample_data'

# test on local machine
ori_path = '/Users/river/sre_sdi_uclproject/ori_data'
resample_path = '/Users/river/sre_sdi_uclproject/resample_data'

# creat folder if not exist
if not os.path.exists(resample_path):
    os.mkdir(resample_path)

# # downsample the original audio to be 8kHz
# def downsample(path):
#     for ori_audio in os.listdir(path):
#         # resample the original to 8kHz
#         resample_audio, _ = librosa.core.load(ori_audio, sr=8000)
#         # save the new audio data
#         np.save(os.path.join(resample_path, ori_audio), resample_audio)
#         print("{} is downsampled".format(ori_audio))

# for ori_audio in os.listdir(ori_path):
#     # resample the original to 8kHz
#     resample_audio, s = librosa.core.load(os.path.join(ori_path, ori_audio), sr=8000)
#     # save the new audio data
#     # np.save(os.path.join(resample_path, ori_audio), resample_audio)
#     meeting = ori_audio.split(".")[0]
#     librosa.output.write_wav(os.path.join(resample_path, meeting), resample_audio, s)
#     print("{} is downsampled".format(ori_audio))

for ori_audio in os.listdir(ori_path):
    meeting = ori_audio.split(".")[0]
    new_name = meeting + ".wav"
    old_path = os.path.join(ori_path, ori_audio)
    new_path = os.path.join(resample_path, new_name)
    subprocess.run(['sox', '-r', '8000', old_path, new_path])

print("Finish")
