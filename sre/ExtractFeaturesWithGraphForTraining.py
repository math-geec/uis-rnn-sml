#!/usr/bin/env python

import glob
from glob import glob
import sys, os, logging
import numpy as np
import math
import scipy.io.wavfile as wav
import scipy.signal
import pathlib
import tensorflow as tf
import pandas as pd


def read_signal(file_name):
    if os.path.isfile(file_name):
        extension = file_name.split('.')[-1]
    if extension == 'wav':
        fs, signal = wav.read(file_name)
        if not fs == 8000:
            logging.info("Unsupported audio format, expected audio input should be 8kHz")
            signal = []
    elif extension == 'raw':
        signal = np.fromfile(file_name, dtype='int16')
    else:
        logging.info('Unvalid file extension, cannot load signal %s', file_name)
        signal = []
    return signal + np.finfo(float).eps


def compute_vad(s, win_length=200, win_overlap=100, n_realignment=5, threshold=0.2):
    F = framing(s, win_length, win_length - win_overlap)
    Energy = 20 * np.log10(np.std(F, axis=1) + np.finfo(float).eps)
    max1 = np.amax(Energy)  # Maximum
    out1 = np.zeros(Energy.shape, dtype=np.bool)
    out1[Energy > max1 - 30] = True
    return out1


def framing(a, window, shift=1):
    shape = (int(np.floor((a.shape[0] - window) / shift + 1)), window) + a.shape[1:]
    strides = (a.strides[0] * shift, a.strides[0]) + a.strides[1:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


class AudioPreprocessing:

    def __init__(self, frozen_model_path):
        self.model = frozen_model_path

        self.load_graph()

        self.session = tf.Session(graph=self.graph, config=None)
        self.input = self.graph.get_operation_by_name("import/audio_in")
        self.output = self.graph.get_operation_by_name("import/audio_features")

    def load_graph(self):
        self.graph = tf.Graph()
        # graph_def = tf.GraphDef()
        graph_def = tf.compat.v1.GraphDef()
        with open(self.model, 'rb') as f:
            graph_def.ParseFromString(f.read())

        with self.graph.as_default():
            tf.import_graph_def(graph_def)

    def extract(self, audio_signal):
        output = self.session.run(self.output.outputs[0], feed_dict={
            self.input.outputs[0]: audio_signal})
        return output


def main():
    # SpkCallList is not necessary here, consider to remove it.
    rootfolder, spkfolder, SpkCallList = sys.argv[1:4]

    if not os.path.exists(spkfolder):
        os.mkdir(spkfolder)

    preproc = AudioPreprocessing("mdl/preprocessing.pb")
    Cond = "Speech"
    for spk in pathlib.Path(rootfolder).iterdir():
        # Iterate through all speakers
        # concatenate all the utterance of same speaker
        # use window of 20s to create same dimension of feature
        spk = str(spk).split('/')[-1]
        print("Processing speaker:", spk)
        FileNameToSave = os.path.join(spkfolder, spk + ".npy")
        if os.path.exists(FileNameToSave) == True:
            continue
        FullUtterance = []
        for audiofile in pathlib.Path(os.path.join(rootfolder, spk)).glob('*.wav'):

            audiofile = str(audiofile)
            logging.info("Processing: %s", audiofile)
            signal = read_signal(audiofile)
            print(audiofile, len(signal))

            if len(signal) < 8000 * 1:
                logging.info("Could not read file %s, or file duration is short. No features will be created",
                             audiofile)
                continue

            fea = preproc.extract(signal)

            vad = compute_vad(signal)
            vad = scipy.signal.medfilt(vad, 11) != 0  # voice parts as True

            if sum(vad) == 0:
                logging.info("File %s does not seem to have speech. skipping!", audiofile)
                continue
            if np.shape(vad)[0] > np.shape(fea)[0]:
                vad = vad[:np.shape(fea)[0]]
            else:
                fea = fea[:np.shape(vad)[0]]

            fea -= np.mean(fea[vad], axis=0)
            speech = fea[vad]

            FullUtterance.append(speech)

        FullUtterance = np.concatenate(FullUtterance)
        IntendedDuration = 2000  # 30 seconds of active speech for segmentation
        NumFiles = int(math.floor((np.shape(FullUtterance)[0] - IntendedDuration) / (IntendedDuration / 2.0))) + 1
        if NumFiles < 1:
            continue
        print(NumFiles, np.shape(FullUtterance))
        utterances_spec = []
        for ii in range(NumFiles):
            utterances_spec.append(
                FullUtterance[int((ii) * IntendedDuration / 2):int((ii) * IntendedDuration / 2 + IntendedDuration)])
        utterances_spec = np.array(utterances_spec)
        utterances_spec = np.einsum('ikj->ijk', utterances_spec)
        if os.path.exists(FileNameToSave):
            utterances_spec_old = np.load(FileNameToSave)
            utterances_spec = np.vstack([utterances_spec_old, utterances_spec])
        np.save(FileNameToSave, utterances_spec)


if __name__ == '__main__':
    main()
