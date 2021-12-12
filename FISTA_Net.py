# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:19:21 2021

@author: Jianhua Guo
"""


'''
Test Platform: Tensorflow version: 1.15.0

Paper Information:

CSI Feedback with Model-Driven Deep Learning of Massive MIMO Systems
Jianhua Guo, Lei Wang, Feng Li, and Jiang Xue

Email: jhguo0525@stu.xjtu.edu.cn
2021/12/12
'''


import tensorflow as tf
import scipy.io as sio
import numpy as np
import tflearn
import os 


tf.reset_default_graph()

envir = 'indoor' #'indoor' or 'outdoor'
CS_ratio        = 32
h_input         = 2048 // CS_ratio
h_output        = 2048
batch_size      = 100
PhaseNumber     = 20
learning_rate   = 1e-4
EpochNum        = 100

## LODA DATA
print('LOAD DATA...')
# Data loading
if envir == 'indoor':
    H_DATA = sio.loadmat('./Data//DATA_Htrainin.mat')
    x_train = H_DATA['HT'].astype(np.float32)                     # H.shape = (None, 32, 16)
    x_train = x_train - 0.5   
     
    mat = sio.loadmat('./Data//DATA_Htestin.mat')
    x_test = mat['HT'] # array
    x_test = x_test - 0.5

elif envir == 'outdoor':
    mat = sio.loadmat('./Data//DATA_Htrainout.mat')
    x_train = mat['HT'] # array
    x_train = x_train - 0.5
    mat = sio.loadmat('./Data//DATA_Htestout.mat')
    x_test = mat['HT'] # array
    x_test = x_test - 0.5


# Initialization    
X_output = tf.placeholder(tf.float32, [None, h_output])
h_0 = tf.zeros_like(X_output)



def res_shrink_block(incoming, 
                     activation='relu', batch_norm=True,
                     bias=True, weights_init='variance_scaling', 
                     bias_init='zeros', regularizer='L2', weight_decay=0.0001, 
                     trainable=True, restore=True, reuse=False, scope=None, 
                     name="ResidualBlock"):
    
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    out_channels = in_channels

    
    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming], reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)
    
    with vscope as scope:
        name = scope.name #TODO

    # get thresholds and apply thresholding
    abs_mean = tf.reduce_mean(tf.reduce_mean(tf.abs(residual),axis=2,keep_dims=True),axis=1,keep_dims=True)
    scales = tflearn.fully_connected(abs_mean, out_channels//4, activation='linear',regularizer='L2',weight_decay=0.0001,weights_init='variance_scaling')
    scales = tflearn.batch_normalization(scales)
    scales = tflearn.activation(scales, 'relu')
    scales = tflearn.fully_connected(scales, out_channels, activation='linear',regularizer='L2',weight_decay=0.0001,weights_init='variance_scaling')
    scales = tf.expand_dims(tf.expand_dims(scales,axis=1),axis=1)
    thres = tf.multiply(abs_mean,tflearn.activations.sigmoid(scales))

    # soft thresholding
    residual = tf.multiply(tf.sign(residual), tf.maximum(tf.abs(residual)-thres,0))

    return residual        

def add_con2d_weight_bias(w_shape, b_shape, order_no):
    Weights = tf.get_variable(shape=w_shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(), name='Weights_%d' % order_no)
    biases = tf.Variable(tf.random_normal(b_shape, stddev=0.05), name='biases_%d' % order_no)
    return [Weights, biases]

def fista_block(codeword, input_layers, W):
    tau_value = tf.Variable(0.01, dtype=tf.float32)
    lambda_step = tf.Variable(0.01, dtype=tf.float32)
    # soft_thr = tf.Variable(0.1, dtype=tf.float32)
    conv_size = 16
    filter_size = 1
    tranW = tf.transpose(W, [1, 0])

    if len(input_layers) == 1:
        x = input_layers[-1]
    else:
        x = tf.add(input_layers[-1], tf.scalar_mul(tau_value, (input_layers[-1] - input_layers[-2])))
  
    Ah = tf.matmul(x, W)
    E = Ah - codeword
    x1_fista = x - tf.scalar_mul(lambda_step, tf.matmul(E, tranW))
    
    x2_fista = tf.reshape(x1_fista, shape=[-1, 32, 32, 2])

    [Weights0, bias0] = add_con2d_weight_bias([3, 3, 2, conv_size], [conv_size], 0)

    [Weights1, bias1] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 1)
    [Weights11, bias11] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 11)

    [Weights2, bias2] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 2)
    [Weights22, bias22] = add_con2d_weight_bias([filter_size, filter_size, conv_size, conv_size], [conv_size], 22)

    [Weights3, bias3] = add_con2d_weight_bias([3, 3, conv_size, 2], [1], 3)

    x3_fista = tf.nn.conv2d(x2_fista, Weights0, strides=[1, 1, 1, 1], padding='SAME')

    x4_fista = tf.nn.relu(tf.nn.conv2d(x3_fista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x44_fista = tf.nn.conv2d(x4_fista, Weights11, strides=[1, 1, 1, 1], padding='SAME')

    # x5_fista = tf.multiply(tf.sign(x44_fista), tf.nn.relu(tf.abs(x44_fista) - soft_thr))
    x5_fista = res_shrink_block(x44_fista)

    x6_fista = tf.nn.relu(tf.nn.conv2d(x5_fista, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x66_fista = tf.nn.conv2d(x6_fista, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x7_fista = tf.nn.conv2d(x66_fista, Weights3, strides=[1, 1, 1, 1], padding='SAME')

    x7_fista = x7_fista + x2_fista

    x8_fista = tf.reshape(x7_fista, shape=[-1, 2048])

    x3_fista_sym = tf.nn.relu(tf.nn.conv2d(x3_fista, Weights1, strides=[1, 1, 1, 1], padding='SAME'))
    x4_fista_sym = tf.nn.conv2d(x3_fista_sym, Weights11, strides=[1, 1, 1, 1], padding='SAME')
    x6_fista_sym = tf.nn.relu(tf.nn.conv2d(x4_fista_sym, Weights2, strides=[1, 1, 1, 1], padding='SAME'))
    x7_fista_sym = tf.nn.conv2d(x6_fista_sym, Weights22, strides=[1, 1, 1, 1], padding='SAME')

    x11_fista = x7_fista_sym - x3_fista

    return [x8_fista, x11_fista]


def autoencoder_fista(X_output, input_tensor, n, reuse):
    
    # 
    W = tf.get_variable(
        'W', 
        [h_output, h_input], 
        tf.float32, 
        initializer=tf.contrib.layers.xavier_initializer())
    
    s = tf.matmul(X_output, W)

    s = tf.reshape(s, shape=[-1, h_input])

    # Decoder
    layers = []
    layers_symetric = []
    layers.append(input_tensor)
    for i in range(n):
        with tf.variable_scope('conv_%d' %i, reuse=reuse):
            [conv1, conv1_sym] = fista_block(s, layers, W)
            layers.append(conv1)
            layers_symetric.append(conv1_sym)
            
    return [layers, layers_symetric]


[Prediction, Pre_symetric] = autoencoder_fista(X_output, h_0, PhaseNumber, reuse=False)

# Loss Function
def compute_cost(Prediction, X_output, PhaseNumber):
    cost = tf.reduce_mean(tf.square(Prediction[-1] - X_output))
    cost_sym = 0
    cost_res = 0

    for k in range(PhaseNumber):
        cost_sym += tf.reduce_mean(tf.square(Pre_symetric[k]))
    
    for kk in range(PhaseNumber-1):
        cost_res += tf.reduce_mean(tf.square(Prediction[kk] - X_output))
        
    cost = cost + 0.01*cost_res
    
    return [cost, cost_sym]


[cost, cost_sym] = compute_cost(Prediction, X_output, PhaseNumber)


cost_all = cost + 0.01 *cost_sym


optm_all = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_all)

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

sess = tf.Session(config=config)
sess.run(init)


print("...............................")
print("Phase Number is %d" % (PhaseNumber))
print("...............................\n")

print("Strart Training..")

# Whether to load pre-trained model
cpkt_model_number = 100
model_dir = 'In_%s_CS_Ration_%d_Phase_%d_ratio_0_Cost2100_FISTA_fc_soft__Net_plus_Model_SNR' % (envir, CS_ratio, PhaseNumber)
saver.restore(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, cpkt_model_number))

output_file_name = "Log_output_%s.txt" % (model_dir)
nrtrain = len(x_train)

# model_dir = 'CS_Ration_%d_Phase_%d_ratio_0_Cost2100_FISTA_fc_soft__Net_plus_Model_SNR' % (CS_ratio, PhaseNumber)


def NMSE_RHO(x, x_hat):
    x     = np.reshape(x, (len(x), 2, 32, 32))
    x_hat = np.reshape(x_hat, (len(x_hat), 2, 32, 32))
    
    x_real = np.reshape(x[:, 0, :, :], (len(x), -1))
    x_imag = np.reshape(x[:, 1, :, :], (len(x), -1))
    # x_C = x_real-0.5 + 1j*(x_imag-0.5)
    x_C = x_real + 1j*(x_imag)
    
    x_hat_real = np.reshape(x_hat[:, 0, :, :], (len(x_hat), -1))
    x_hat_imag = np.reshape(x_hat[:, 1, :, :], (len(x_hat), -1))
    # x_hat_C = x_hat_real-0.5 + 1j*(x_hat_imag-0.5)
    x_hat_C = x_hat_real + 1j*(x_hat_imag)
    
    # x_hat_F = np.reshape(x_hat_C, (len(x_hat_C), 32, 32))
    # X_hat = np.fft.fft(np.concatenate((x_hat_F, np.zeros((len(x_hat_C), 32, 257-32))), axis=2), axis=2)
    # X_hat = X_hat[:, :, 0:125]
    
    # n1 = np.sqrt(np.sum(np.conj(x)*x, axis=1))
    # n1 = n1.astype('float64')
    # n2 = np.sqrt(np.sum(np.conj(X_hat)*X_hat, axis=1))
    # n2 = n2.astype('float64')
    # aa = abs(np.sum(np.matmul(np.conj(x), X_hat), axis=1))
    # rho = np.mean(aa/(n1*n2), axis=1)
    # rho = 1
    power = np.sum(abs(x_C)**2, axis=1)
    mse = np.sum(abs(x_C-x_hat_C)**2, axis=1)

    return 10*np.log10(np.mean(mse/power))


for epoch_i in range(EpochNum+1):
    randidx_all = np.random.permutation(len(x_train))
    for batch_i in range(nrtrain // batch_size):
        randidx = randidx_all[batch_i*batch_size:(batch_i+1)*batch_size]

        batch_h    = x_train[randidx, :]

        feed_dict = {X_output: batch_h}
        sess.run(optm_all, feed_dict=feed_dict)

    pre_h = sess.run(Prediction[-1], feed_dict=feed_dict)
    NMSE = NMSE_RHO(batch_h, pre_h)

    output_data = "[%02d/%02d] cost: %.4f, cost_sym: %.4f , NMSE: %.4f \n" % (
        epoch_i, EpochNum, sess.run(cost, feed_dict=feed_dict), 
        sess.run(cost_sym, feed_dict=feed_dict), NMSE)
    
    print(output_data)
    
    output_file = open(output_file_name, 'a')
    output_file.write(output_data)
    output_file.close()

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if epoch_i <= 30:
        saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)
    else:
        if epoch_i % 20 == 0:
            saver.save(sess, './%s/CS_Saved_Model_%d.cpkt' % (model_dir, epoch_i), write_meta_graph=False)


print("Training Finished")
