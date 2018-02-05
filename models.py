import tensorflow as tf
import numpy as np
from plot_util import *

class Loss:
    BINARY_CE, RANGE0_255, RANGE_MEAN = range(3)

def get_loss(loss_type, y_,y):
    if(loss_type==Loss.BINARY_CE):
        cost_per_pixel = y_*tf.log(y+1E-13) + (1-y_)*tf.log(1-y+1E-13)  #add +1E-13 to prevent log(0)
        return -1*tf.reduce_mean(cost_per_pixel)

class SimpleModel():
    def __init__(self,shape,filter_size,num_filters,loss_type):

        self.input = tf.placeholder(tf.float32, [None, shape[0], shape[1], 1], 'input_images')
        self.segm_map = tf.placeholder(tf.float32, [None, shape[0], shape[1]], 'output_segm_map')
        self.dropout = tf.placeholder(dtype=tf.float32,name='dropout')

        W1 = tf.get_variable('weight1', [filter_size[0], filter_size[1], 1, num_filters[0]])
        b1 = tf.get_variable('bias1',[num_filters[0],])
        a1 = tf.nn.conv2d(self.input, W1,[1,1,1,1], "SAME")+ b1
        h1 = tf.nn.relu(tf.nn.dropout(a1, self.dropout))

        W2 = tf.get_variable('weight2', [filter_size[0], filter_size[1], num_filters[0], num_filters[1]])
        b2 = tf.get_variable('bias2',[num_filters[1],])
        a2 = tf.nn.conv2d(h1, W2,[1,1,1,1], "SAME")+ b2
        h2 = tf.nn.relu(tf.nn.dropout(a2, self.dropout))

        W3 = tf.get_variable('weight3', [filter_size[0], filter_size[1], num_filters[1], num_filters[2]])
        b3 = tf.get_variable('bias3',[num_filters[2],])
        a3 = tf.nn.conv2d(h2, W3,[1,1,1,1], "SAME")+ b3
        h3 = tf.nn.relu(tf.nn.dropout(a3, self.dropout))

        W4 = tf.get_variable('weight4', [filter_size[0], filter_size[1], num_filters[2], num_filters[3]])
        b4 = tf.get_variable('bias4',[num_filters[3],])
        a4 = tf.nn.conv2d(h3, W4,[1,1,1,1], "SAME")+ b4
        h4 = tf.nn.relu(tf.nn.dropout(a4, self.dropout))

        W5 = tf.get_variable('weight5', [filter_size[0], filter_size[1], num_filters[3], 1])
        b5 = tf.get_variable('bias5',[1,])
        a5 = tf.squeeze(tf.nn.conv2d(h4, W5,[1,1,1,1], "SAME")+ b5, 3)
        self.h5 = tf.nn.sigmoid(a5)
        self.loss= get_loss(loss_type,self.segm_map,self.h5)
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)


    def train(self,sess,data,batch_size=5,iterations = 1000,dropout=0.7,info_step=50):
        for iter in range(iterations):
            x, y_ = data.next_batch(batch_size)

            if iter%info_step == 0:
                #plot_image(x[2,:,:,0])
                #plot_image(y_[2])
                train_loss, segm_map_pred= sess.run([self.loss, self.h5], feed_dict={self.input:x, self.segm_map:y_, self.dropout:dropout})
                #plot_image(segm_map_pred[2])
                print('iter %5i/%5i loss is %5.3f and mIOU %5.3f'%(iter, iterations, train_loss, calc_iou(y_, segm_map_pred)))

            train_loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.input:x, self.segm_map:y_, self.dropout:dropout})

    def test(self,sess,test_data):

        return sess.run([self.loss, self.h5], feed_dict={self.input:test_data.images, self.segm_map:test_data.masks, self.dropout:1})


class u_net:
    def __init__(self,shape,filter_size,num_filters,loss_type):
        self.input = tf.placeholder(tf.float32, [None, shape[0], shape[1], 1], 'input_images')
        self.segm_map = tf.placeholder(tf.float32, [None, shape[0], shape[1]], 'output_segm_map')
        self.dropout = tf.placeholder(dtype=tf.float32,name='dropout')

        batch_size = tf.shape(self.input)[0]
        nx = shape[0]
        ny = shape[1]
        nz = 1

        W1_1 = tf.get_variable('weight1_1', [filter_size[0], filter_size[1], 1, num_filters[0]])
        b1_1 = tf.get_variable('bias1_1',[num_filters[0],])
        a1_1 = tf.nn.conv2d(self.input, W1_1,[1,1,1,1], "SAME")+ b1_1
        h1_1 = tf.nn.relu(tf.nn.dropout(a1_1, self.dropout))

        W1_2 = tf.get_variable('weight1_2', [filter_size[0], filter_size[1], num_filters[0], num_filters[0]])
        b1_2 = tf.get_variable('bias1_2',[num_filters[0],])
        a1_2 = tf.nn.conv2d(h1_1, W1_2,[1,1,1,1], "SAME")+ b1_2
        h1_2 = tf.nn.max_pool(tf.nn.relu(a1_2),[1,2,2,1],[1,2,2,1], "SAME")

        W2_1 = tf.get_variable('weight2_1', [filter_size[0], filter_size[1], num_filters[0], num_filters[1]])
        b2_1 = tf.get_variable('bias2_1',[num_filters[1],])
        a2_1 = tf.nn.conv2d(h1_2, W2_1,[1,1,1,1], "SAME")+ b2_1
        h2_1 = tf.nn.relu(tf.nn.dropout(a2_1, self.dropout))

        W2_2 = tf.get_variable('weight2_2', [filter_size[0], filter_size[1], num_filters[1], num_filters[1]])
        b2_2 = tf.get_variable('bias2_2',[num_filters[1],])
        a2_2 = tf.nn.conv2d(h2_1, W2_2,[1,1,1,1], "SAME")+ b2_2
        h2_2 = tf.nn.max_pool(tf.nn.relu(a2_2),[1,2,2,1],[1,2,2,1], "SAME")

        W3_1 = tf.get_variable('weight3_1', [filter_size[0], filter_size[1], num_filters[1], num_filters[2]])
        b3_1 = tf.get_variable('bias3_1',[num_filters[2],])
        a3_1 = tf.nn.conv2d(h2_2, W3_1,[1,1,1,1], "SAME")+ b3_1
        h3_1 = tf.nn.relu(tf.nn.dropout(a3_1, self.dropout))

        W3_2 = tf.get_variable('weight3_2', [filter_size[0], filter_size[1], num_filters[2], num_filters[2]])
        b3_2 = tf.get_variable('bias3_2',[num_filters[2],])
        a3_2 = tf.nn.conv2d(h3_1, W3_2,[1,1,1,1], "SAME")+ b3_2
        h3_2 = tf.nn.max_pool(tf.nn.relu(a3_2),[1,2,2,1],[1,2,2,1], "SAME")

        W4_1 = tf.get_variable('weight4_1', [filter_size[0], filter_size[1], num_filters[2], num_filters[3]])
        b4_1 = tf.get_variable('bias4_1',[num_filters[3],])
        a4_1 = tf.nn.conv2d(h3_2, W4_1,[1,1,1,1], "SAME")+ b4_1
        h4_1 = tf.nn.relu(tf.nn.dropout(a4_1, self.dropout))

        W4_2 = tf.get_variable('weight4_2', [filter_size[0], filter_size[1], num_filters[3], num_filters[3]])
        b4_2 = tf.get_variable('bias4_2',[num_filters[3],])
        a4_2 = tf.nn.conv2d(h4_1, W4_2,[1,1,1,1], "SAME")+ b4_2
        h4_2 = tf.nn.max_pool(tf.nn.relu(a4_2),[1,2,2,1],[1,2,2,1], "SAME")

        W5_1 = tf.get_variable('weight5_1', [filter_size[0], filter_size[1], num_filters[3], num_filters[4]])
        b5_1 = tf.get_variable('bias5_1',[num_filters[4],])
        a5_1 = tf.nn.conv2d(h4_2, W5_1,[1,1,1,1], "SAME")+ b5_1
        h5_1 = tf.nn.relu(tf.nn.dropout(a5_1, self.dropout))

        W5_2 = tf.get_variable('weight5_2', [filter_size[0], filter_size[1], num_filters[4], num_filters[4]])
        b5_2 = tf.get_variable('bias5_2',[num_filters[4],])
        a5_2 = tf.nn.conv2d(h5_1, W5_2,[1,1,1,1], "SAME")+ b5_2
        h5_2 = tf.nn.relu(tf.nn.dropout(a5_2, self.dropout))

        Wc4_1 = tf.get_variable('weightc4_1', [filter_size[0], filter_size[1], num_filters[3], num_filters[4]])
        bc4_1 = tf.get_variable('biasc4_1',[num_filters[3],])
        ac4_1 = tf.nn.conv2d_transpose(h5_2,Wc4_1,[batch_size,nx//8,ny//8,512],[1,2,2,1], "SAME")+ bc4_1
        hc4_1 = tf.concat([a4_2,tf.nn.relu(ac4_1)],3)

        Wc4_2 = tf.get_variable('weightc4_2', [filter_size[0], filter_size[1], num_filters[4], num_filters[3]])
        bc4_2 = tf.get_variable('biasc4_2',[num_filters[3],])
        ac4_2 = tf.nn.conv2d(hc4_1, Wc4_2,[1,1,1,1], "SAME")+ bc4_2
        hc4_2 = tf.nn.relu(ac4_2)

        Wc4_3 = tf.get_variable('weightc4_3', [filter_size[0], filter_size[1], num_filters[3], num_filters[3]])
        bc4_3 = tf.get_variable('biasc4_3',[num_filters[3],])
        ac4_3 = tf.nn.conv2d(hc4_2, Wc4_3,[1,1,1,1], "SAME")+ bc4_3
        hc4_3 = tf.nn.relu(tf.nn.dropout(ac4_3, self.dropout))

        Wc3_1 = tf.get_variable('weightc3_1', [filter_size[0], filter_size[1], num_filters[2], num_filters[3]])
        bc3_1 = tf.get_variable('biasc3_1',[num_filters[2],])
        ac3_1 = tf.nn.conv2d_transpose(hc4_3,Wc3_1,[batch_size,nx//4,ny//4,256],[1,2,2,1], "SAME")+ bc3_1
        hc3_1 = tf.concat([a3_2,tf.nn.relu(ac3_1)],3)

        Wc3_2 = tf.get_variable('weightc3_2', [filter_size[0], filter_size[1], num_filters[3], num_filters[2]])
        bc3_2 = tf.get_variable('biasc3_2',[num_filters[2],])
        ac3_2 = tf.nn.conv2d(hc3_1, Wc3_2,[1,1,1,1], "SAME")+ bc3_2
        hc3_2 = tf.nn.relu(ac3_2)

        Wc3_3 = tf.get_variable('weightc3_3', [filter_size[0], filter_size[1], num_filters[2], num_filters[2]])
        bc3_3 = tf.get_variable('biasc3_3',[num_filters[2],])
        ac3_3 = tf.nn.conv2d(hc3_2, Wc3_3,[1,1,1,1], "SAME")+ bc3_3
        hc3_3 = tf.nn.relu(tf.nn.dropout(ac3_3, self.dropout))

        Wc2_1 = tf.get_variable('weightc2_1', [filter_size[0], filter_size[1], num_filters[1], num_filters[2]])
        bc2_1 = tf.get_variable('biasc2_1',[num_filters[1],])
        ac2_1 = tf.nn.conv2d_transpose(hc3_3,Wc2_1,[batch_size,nx//2,ny//2,128],[1,2,2,1], "SAME")+ bc2_1
        hc2_1 = tf.concat([a2_2,tf.nn.relu(ac2_1)],3)

        Wc2_2 = tf.get_variable('weightc2_2', [filter_size[0], filter_size[1], num_filters[2], num_filters[1]])
        bc2_2 = tf.get_variable('biasc2_2',[num_filters[1],])
        ac2_2 = tf.nn.conv2d(hc2_1, Wc2_2,[1,1,1,1], "SAME")+ bc2_2
        hc2_2 = tf.nn.relu(ac2_2)

        Wc2_3 = tf.get_variable('weightc2_3', [filter_size[0], filter_size[1], num_filters[1], num_filters[1]])
        bc2_3 = tf.get_variable('biasc2_3',[num_filters[1],])
        ac2_3 = tf.nn.conv2d(hc2_2, Wc2_3,[1,1,1,1], "SAME")+ bc2_3
        hc2_3 = tf.nn.relu(tf.nn.dropout(ac2_3, self.dropout))

        Wc1_1 = tf.get_variable('weightc1_1', [filter_size[0], filter_size[1], num_filters[0], num_filters[1]])
        bc1_1 = tf.get_variable('biasc1_1',[num_filters[0],])
        ac1_1 = tf.nn.conv2d_transpose(hc2_3,Wc1_1,[batch_size,nx,ny,64],[1,2,2,1], "SAME")+ bc1_1
        hc1_1 = tf.concat([a1_2,tf.nn.relu(ac1_1)],3)

        Wc1_2 = tf.get_variable('weightc1_2', [filter_size[0], filter_size[1], num_filters[1], num_filters[0]])
        bc1_2 = tf.get_variable('biasc1_2',[num_filters[0],])
        ac1_2 = tf.nn.conv2d(hc1_1, Wc1_2,[1,1,1,1], "SAME")+ bc1_2
        hc1_2 = tf.nn.relu(ac1_2)

        Wc1_3 = tf.get_variable('weightc1_3', [filter_size[0], filter_size[1], num_filters[0], num_filters[0]])
        bc1_3 = tf.get_variable('biasc1_3',[num_filters[0],])
        ac1_3 = tf.nn.conv2d(hc1_2, Wc1_3,[1,1,1,1], "SAME")+ bc1_3
        hc1_3 = tf.nn.relu(tf.nn.dropout(ac1_3, self.dropout))
             
        W0 = tf.get_variable('weight0', [filter_size[0], filter_size[1], num_filters[0], 1])
        b0 = tf.get_variable('bias0',[1,])
        a0 = tf.squeeze(tf.nn.conv2d(hc1_3, W0,[1,1,1,1], "SAME")+ b0, 3)
        self.h0 = tf.nn.sigmoid(a0)

        self.loss= get_loss(loss_type,self.segm_map,self.h0)
        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def train(self,sess,data,batch_size=5,iterations = 1000,dropout=0.7,info_step=50):
        for iter in range(iterations):
            x, y_ = data.next_batch(batch_size)

            if iter%info_step == 0:
                #plot_image(x[2,:,:,0])
                #plot_image(y_[2])
                train_loss, segm_map_pred= sess.run([self.loss, self.h0], feed_dict={self.input:x, self.segm_map:y_, self.dropout:dropout})
                #plot_image(segm_map_pred[2])
                print('iter %5i/%5i loss is %5.3f and mIOU %5.3f'%(iter, iterations, train_loss, calc_iou(y_, segm_map_pred)))

            train_loss, _ = sess.run([self.loss, self.train_step], feed_dict={self.input:x, self.segm_map:y_, self.dropout:dropout})
    def test(self,sess,test_data):

        return sess.run([self.loss, self.h0], feed_dict={self.input:test_data.images, self.segm_map:test_data.masks, self.dropout:1})