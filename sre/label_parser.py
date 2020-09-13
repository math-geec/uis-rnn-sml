import os
import numpy as np
import xml.etree.ElementTree as ET

################################
# Get speaker label from DialogueAct.xml file
# Each speaker has an unique id
# Training id should be in form 'a_b' as an array of string,
# where a = meeting id, b = speaker id
################################

# def label_parser(meeting, label):
#     prefixed = [filename for filename in os.listdir('/Users/river/sre_sdi_uclproject/dialogue') if filename.startswith(meeting)]
#     tree = ET.parse('/Users/river/sre_sdi_uclproject/dialogue/Bro015.D.dialogue-acts.xml')
#     root = tree.getroot()
#     print('root.attrib: ', root.attrib)
#     meeting = root.attrib[0]
#
#     for dialogueact in root.iter('dialogueact'):
#         print(dialogueact.attrib)
#         participant = dialogueact.get('participant')
#         start_time = dialogueact.get('starttime')
#         end_time = dialogueact.get('endtime')

tree = ET.parse('/Users/river/sre_sdi_uclproject/dialogue/Bro015.D.dialogue-acts.xml')
root = tree.getroot()
print('root.attrib: ', root.attrib)
# meeting = root.attrib[0].split(".")[0]
label = np.empty(812, dtype=object)

# get speaker id
speaker_id = root.find('dialogueact').get('participant')
# add meeting id and speaker id
speaker = meeting + '_' + speaker_id

# get start and end time
for dialogueact in root.iter('dialogueact'):
    # print(dialogueact.attrib)
    # participant = dialogueact.get('participant')
    start_time = int(round(float(dialogueact.get('starttime')) * 0.8))
    end_time = int(round(float(dialogueact.get('endtime')) * 0.8))
    # add speaker id to label
    while start_time < end_time:
        label[start_time - 1] = speaker
        start_time += 1

print("pass")
