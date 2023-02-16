import tensorflow as tf
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

event_file_path = "masterthesis_test/CBF_logs/run1_with_OptNet/lightning_logs/version_0/events.out.tfevents.1676305934.wangxinyu-TUF-Gaming-FX505GE-FX86FE.45358.0"


runlog = pd.DataFrame(columns=['epoch', 'loss', 'loss_type'])

step = 0
count = 0 
current_epoch = 3.0
for e in tf.train.summary_iterator(event_file_path):
    
    r = {}
    
    for v in e.summary.value:
        if v.tag == 'epoch':
            current_epoch = v.simple_value

        if v.tag == 'Total_loss/train' or v.tag == 'Total_loss/val':
            # print(f"v.tag is {v.tag}")
            # print(f"v.simple_value is {v.simple_value} \n")
            r['loss'] = v.simple_value
            r['loss_type'] = v.tag
            r['epoch'] = current_epoch
    
            runlog = runlog.append(r, ignore_index=True)
        
    

# print(runlog.columns)
# print(runlog.index)
print(runlog)

sns.set_theme()

sns.relplot(data=runlog, x='epoch', y='loss', hue='loss_type',kind='line')
plt.show()





# #!/usr/bin/env python3

# '''
# This script exctracts training variables from all logs from 
# tensorflow event files ("event*"), writes them to Pandas 
# and finally stores in long-format to a CSV-file including
# all (readable) runs of the logging directory.
# The magic "5" infers there are only the following v.tags:
# [lr, loss, acc, val_loss, val_acc]
# '''

# import tensorflow as tf
# import glob
# import os
# import pandas as pd


# # Get all event* runs from logging_dir subdirectories
# logging_dir = './logs'
# event_paths = glob.glob(os.path.join(logging_dir, "*","event*"))


# # Extraction function
# def sum_log(path):
#     runlog = pd.DataFrame(columns=['metric', 'value'])
#     try:
#         for e in tf.train.summary_iterator(path):
#             for v in e.summary.value:
#                 r = {'metric': v.tag, 'value':v.simple_value}
#                 runlog = runlog.append(r, ignore_index=True)
    
#     # Dirty catch of DataLossError
#     except:
#         print('Event file possibly corrupt: {}'.format(path))
#         return None

#     runlog['epoch'] = [item for sublist in [[i]*5 for i in range(0, len(runlog)//5)] for item in sublist]
    
#     return runlog


# # Call & append
# all_log = pd.DataFrame()
# for path in event_paths:
#     log = sum_log(path)
#     if log is not None:
#         if all_log.shape[0] == 0:
#             all_log = log
#         else:
#             all_log = all_log.append(log)


# # Inspect
# print(all_log.shape)
# all_log.head()    
            
# # Store
# all_log.to_csv('all_training_logs_in_one_file.csv', index=None)