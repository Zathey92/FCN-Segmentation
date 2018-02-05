import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from data_util import DataSet,Normalization
from models import *

dataset = DataSet(256,Normalization.RANGE_MEAN)
filenames = ['LIDC-0001','LIDC-0836']
margins = [[29,0],[33,0]]
dataset.load_filenames(filenames,10,margins)
#model = SimpleModel([256,256],[5,5],[8, 32,32, 16],Loss.BINARY_CE)
model = u_net([256,256],[3,3],[64,128, 256,512,1024],Loss.BINARY_CE)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
print('------------TRAIN------------')
model.train(sess,dataset.train,4,3000)
loss,segm_map_pred=model.test(sess,dataset.test)
print('------------TEST-------------')
print('Loss is %5.3f and mIOU %5.3f'%(loss, calc_iou(dataset.test.masks, segm_map_pred)))
plot_segm_map(np.reshape(dataset.test.images,[len(dataset.test.images),256,256]), dataset.test.masks, segm_map_pred)

