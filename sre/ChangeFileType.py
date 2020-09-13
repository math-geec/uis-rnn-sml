import os
import subprocess
import numpy as np

# Original type for voxceleb dataset is ".m4a"
# Change it to ".wav" type for further preprocessing

directory = "/uclprojects/SpeakerRecognition/dev/aac"
f = open("Voxceleb.lst", 'a')
speakers = os.listdir(directory)

for spk in speakers:
    new_path = os.path.join(directory, spk)
    video_ids = os.listdir(new_path)

    for vid in video_ids:
        old_path = os.path.join(new_path, vid)
        utterance = os.listdir(old_path)

        for utter in utterance:
            name = utter.split('.')[0]
            new_name = name + ".wav"
            subprocess.call(
                ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-i', os.path.join(old_path, utter), '-ac', '1',
                 '-ar', '8000', '-acodec', 'pcm_s16le', os.path.join(new_path, new_name)])
            f.write(spk + '\t' + name + '\n')
            os.remove(os.path.join(old_path, utter))

        os.rmdir(old_path)
