# import librosa
import tensorflow as tf
import numpy as np
import os
# import time
import scipy
import random
# from scipy.sparse import dia_matrix
# from itertools import combinations
from utils import random_batch, normalize, similarity, loss_cal, optim
from configuration import get_config
# from tensorflow.contrib import rnn
from sklearn import preprocessing
import xml.etree.ElementTree as ET
from ExtractFeaturesWithGraphForTraining import read_signal, compute_vad, framing, AudioPreprocessing


def convert_to_feat(audio):
    signal = read_signal(audio)
    fea = preproc.extract(signal)
    vad = compute_vad(signal)
    vad = scipy.signal.medfilt(vad, 11) != 0  # voice parts as True
    if np.shape(vad)[0] > np.shape(fea)[0]:
        vad = vad[:np.shape(fea)[0]]
    else:
        fea = fea[:np.shape(vad)[0]]
    # fea=fea[vad]
    fea = fea - np.mean(fea, axis=0)
    return fea


# config = get_config()

# extract embedding from features with certain window size
def extract_embeddings(feature, window):
    embeddings = np.empty((0, 64))
    for i in range(0, feature.shape[0], window):
        if i + window <= feature.shape[0]:
            sub_feature = feature[i:i + window, :]
        else:
            sub_feature = feature[i:, :]
        one_embedding = sess.run(outputfun, feed_dict={inputtensor: np.einsum('jik->ijk', [sub_feature])})
        embeddings = np.vstack((embeddings, one_embedding))
    return embeddings


# Get speaker label from DialogueAct.xml file
def label_parser(meeting, label, dia_act):
    tree = ET.parse(os.path.join(config.dialogue_path, dia_act))
    root = tree.getroot()
    # print('root.attrib: ', root.attrib)

    # get speaker id
    speaker_id = root.find('dialogueact').get('participant')
    # add meeting id and speaker id
    speaker = meeting + '_' + speaker_id

    # get start and end time
    for dialogueact in root.iter('dialogueact'):
        # print(dialogueact.attrib)
        start_time = int(round(float(dialogueact.get('starttime'))))
        end_time = int(round(float(dialogueact.get('endtime'))))
        # add speaker id to label
        while start_time <= end_time:
            label[start_time - 1] = speaker
            start_time += 1
    return label


print("\nTest session")
config = get_config()
preproc = AudioPreprocessing("mdl/preprocessing.pb")

# test np.load
# embedding = np.load(os.path.join(config.embedding_path, 'Bro015.embedding.npy'))
demo_embedding = np.load(os.path.join(config.embedding_path, 'Bro015.embedding.npy'))
demo_label = np.load(os.path.join(config.label_path, 'Bro015.label.npy'), allow_pickle=True)

# build test data
window = 30
final_emb = []
final_label = []
for i in range(0, demo_embedding.shape[0], window):
    if i + window <= demo_embedding.shape[0]:
        sub_emb = demo_embedding[i:i + window, :]
        sub_list = demo_label[i:i + window].tolist()
    else:
        sub_emb = demo_embedding[i:, :]
        sub_list = demo_label[i:].tolist()
    final_emb.append(sub_emb)
    final_label.append(sub_list)

tf.reset_default_graph()

# draw graph
inputtensor = tf.placeholder(shape=[None, 1, 20], dtype=tf.float32)  # enrollment batch (time x batch x n_mel)

# embedding lstm (3-layer default)
with tf.variable_scope("lstm"):
    lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in
                  range(config.num_layer)]
    lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)  # make lstm op and variables
    outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=inputtensor, dtype=tf.float32,
                                   time_major=True)  # for TI-VS must use dynamic rnn
    embedded = outputs[-1]  # the last ouput is the embedded d-vector
    embedded = normalize(embedded)  # normalize

outputfun = 1 * embedded

saver = tf.train.Saver(var_list=tf.global_variables())
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # load model
    # print("model path :", config.model_path)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(config.model_path, "Check_Point"))
    ckpt_list = ckpt.all_model_checkpoint_paths
    loaded = 0
    for model in ckpt_list:
        if config.model_num == int(model.split('-')[-1]):  # find ckpt file which matches configuration model number
            # print("ckpt file is loaded !", model)
            loaded = 1
            saver.restore(sess, model)  # restore variables from selected ckpt file
            break

    if loaded == 0:
        raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")
    # Dictionary for speakers 
    # spk_dict = generate_spk_dict(config.test_path)

    # Convert all the audio files to feature embedding
    if os.path.exists(config.embedding_path) == False:
        os.mkdir(config.embedding_path)
    if os.path.exists(config.label_path) == False:
        os.mkdir(config.label_path)
    # for spk in os.listdir(config.test_path):
    #     if os.path.isdir(os.path.join(config.embedding_path,spk)) == False:
    #         os.mkdir(os.path.join(config.embedding_path,spk))
    #     for vid in os.listdir(os.path.join(config.test_path,spk)):
    #         if os.path.isdir(os.path.join(config.embedding_path,spk,vid)) == False:
    #             os.mkdir(os.path.join(config.embedding_path,spk,vid))
    #         for utter in os.listdir(os.path.join(config.test_path,spk,vid)):
    #             name = utter.split(".")[0]
    #             if os.path.isfile(os.path.join(config.embedding_path,spk,vid,name+'.npy')) == False:
    #                 print("converting ",os.path.join(spk,utter))
    #                 feat = convert_to_feat(os.path.join(config.test_path,spk,vid,utter))
    #                 embedding = sess.run(outputfun,feed_dict={inputtensor: np.einsum('jik->ijk', [feat])})
    #                 np.save(os.path.join(config.embedding_path,spk,vid,name+'.npy'), embedding)

    for resample_audio in os.listdir(config.test_path):
        meeting = resample_audio.split(".")[0]
        if os.path.isfile(os.path.join(config.embedding_path, meeting + '.embedding.npy')) == False:
            print("converting {}".format(resample_audio))
            # resample the original to 8kHz
            # resample_audio, _ = librosa.core.load(os.path.join(config.test_path, ori_audio), sr=8000)
            # convert audio to features
            # for every 10msec of speech a feature vector with dimensionality of 20 is extracted
            feat = convert_to_feat(os.path.join(config.test_path, resample_audio))
            # embedding = sess.run(outputfun,feed_dict={inputtensor: np.einsum('jik->ijk', [feat])})
            # extract embedding every 1s, i.e. every 80 features
            embedding = extract_embeddings(feat, 80)
            np.save(os.path.join(config.embedding_path, meeting + '.embedding.npy'), embedding)
        # initialize label array with label
        # label = np.empty(embedding.shape[0], dtype=object)
        # initialize label array with label "x", which indicates unknown such as silence, background noise or missing timestamps
        label = np.array(['x' for _ in range(embedding.shape[0])], dtype="<U15")
        # find all dialog acts files for the related meeting
        prefixed = [dia for dia in os.listdir(config.dialogue_path) if dia.split(".")[0] == meeting]
        # build speaker label
        for dia_act in prefixed:
            label = label_parser(meeting, label, dia_act)
        # conv = lambda i: i or 'x'
        # label = [conv(i) for i in label]
        np.save(os.path.join(config.label_path, meeting + '.label.npy'), label)

    # Calculate equal error rate
    true_list = []
    false_list = []
    trial_list = open(config.trial_pair, 'r').readlines()
    print('Processing {} verification tests'.format(len(trial_list)))
    for pair in trial_list:
        label, enrol, verif = pair.split()
        enrol = np.load(os.path.join(config.embedding_path, enrol.split('.')[0] + '.npy'))
        verif = np.load(os.path.join(config.embedding_path, verif.split('.')[0] + '.npy'))
        cos_sim = np.dot(enrol, verif.T) / np.linalg.norm(enrol) / np.linalg.norm(verif)
        if label == '0':
            false_list.append(cos_sim)
        else:
            true_list.append(cos_sim)

    diff = 1;
    EER = 0;
    EER_thres = 0;
    EER_FAR = 0;
    EER_FRR = 0
    for thres in [0.01 * i for i in range(100)]:
        FAR = len([i for i in false_list if i > thres]) / len(false_list)
        FRR = len([i for i in true_list if i < thres]) / len(true_list)
        # Save threshold when FAR = FRR (=EER)
        if diff > abs(FAR - FRR):
            diff = abs(FAR - FRR)
            EER = (FAR + FRR) / 2
            EER_thres = thres
            EER_FAR = FAR
            EER_FRR = FRR

    print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" % (EER, EER_thres, EER_FAR, EER_FRR))

    # Make a directory to save all enrolment embeddings
    # and remove utterance used to compute enrolment embeddings
    enrol_path = os.path.join(config.embedding_path, 'enrolment')
    if os.path.exists(enrol_path) == False:
        os.mkdir(enrol_path)
    # list of speaker with more than 5 utterance. 4 will be used for enrolment
    # at least 1 will be used for identification
    spk_list = [spk for spk in os.listdir(config.test_path) if len(
        [os.path.join(r, file) for r, d, f in os.walk(os.path.join(config.test_path, spk)) for file in f]) >= 5]
    for spk in spk_list:
        print("Enrolment for speaker ", spk)
        files = [os.path.join(r, file) for r, d, f in os.walk(os.path.join(config.test_path, spk)) for file in f]
        utter_sample = random.sample(files, 4)
        enrol_feat = []
        for utter in utter_sample:
            enrol_feat.append(convert_to_feat(utter))
            old_ids = utter.split('/')
            new_ids = old_ids[-3:]
            new_ids.insert(0, config.embedding_path)
            new_ids[-1] = new_ids[-1][:-4] + '.npy'
            os.remove('/'.join(new_ids))
        enrol_embedding = sess.run(outputfun,
                                   feed_dict={inputtensor: np.einsum('jik->ijk', [np.concatenate(enrol_feat)])})
        np.save(os.path.join(enrol_path, spk + '.npy'), enrol_embedding)

    # Calculate identification accuracy
    print("Calculating identification accuracy")
    test_num = 0
    correct_num = 0

    spks = os.listdir(enrol_path)
    enrol_list = []
    for file in spks:
        enrol_list.append(np.load(os.path.join(enrol_path, file)))
    enrol_mat = np.concatenate(enrol_list)
    enrol_mat = preprocessing.normalize(enrol_mat, axis=1, norm='l2')
    for spk in spk_list:
        print("Identification test on speaker", spk)
        tst_path = os.path.join(config.embedding_path, spk)
        true_pos = spks.index(spk + '.npy')
        for vid in os.listdir(tst_path):
            test_num += len(os.listdir(os.path.join(tst_path, vid)))
            for utter in os.listdir(os.path.join(tst_path, vid)):
                tst = np.load(os.path.join(tst_path, vid, utter))
                pred = np.dot(enrol_mat, tst.T) / np.linalg.norm(tst)
                if np.argmax(pred) == true_pos:
                    correct_num += 1

    print("Identification accuracy:", correct_num / test_num)
