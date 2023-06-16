import os
import glob


normal_file_list = glob.glob('CF-FA/NORMAL/*-*.jpg')
abnormal_file_list = glob.glob('CF-FA/ABNORMAL/*-*.jpg')

'''
for file in normal_file_list:
    other_file = file.split('-')[-1]
    other_file = 'CF-FA/NORMAL/' + other_file
    print(file, other_file)
    command = 'gdbicp ' + other_file + ' ' + file + ' -model 0 -complete -invert -xform_as_to'
    os.system(command)
'''

for file in abnormal_file_list:
    other_file = file.split('-')[-1]
    other_file = 'CF-FA/ABNORMAL/' + other_file
    print(file, other_file)
    command = 'gdbicp ' + file + ' ' + other_file + ' -model 0 -complete -invert -xform_as_to'
    os.system(command)

