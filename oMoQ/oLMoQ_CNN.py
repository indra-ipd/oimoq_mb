import tensorflow as tf
print(tf.__version__)

import collections
import time
import matplotlib.pyplot as plt
import numpy as np

'''MNISTデータの呼び出し'''
(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()
Xtrain = Xtrain.reshape(-1,784)
Xtest = Xtest.reshape(-1,784)
Ytrain = tf.keras.utils.to_categorical(Ytrain,num_classes=10)
Ytest = tf.keras.utils.to_categorical(Ytest,num_classes=10)

#seed setting
seed = 100
print(seed)
tf.set_random_seed(seed)

#実験条件
image_size = 28
chan_num = 1
labels_size = 10
learning_rate = 0.01
batch_size = 120
batches = int(len(Ytrain) / batch_size)#60000/129
epoch = 10

#iterations : グラフ作成　epごとに初期化されない
iterations = batches
iterations_mb=batches*3



#アルゴリズム一覧
algo = ['oLMoQ_mb','oLNAQ','oLMoQ']#'oLBFGS1','oLBFGS2',,,'oLMoQ_1',,'Adam'
col = {'oLNAQ': 'b', 'oLMoQ': 'm', 'oLMoQ_1': 'k', 'oLBFGS1': 'g', 'oLMoQ_mb': 'y', 'oLBFGS2': 'y', 'Adam': 'r'}

'''
ミニバッチ作成
x_tr : 訓練データ
y_tr　：ラベル
size : ミニバッチのサイズ

num_batch : ミニバッチの数
data : 訓練データ[num_batch, size, 784]
lab : 訓練ラベル[num_batch, size, 784]
'''
def get_batches(x_tr, y_tr, size):
    num_batch = int(len(y_tr) / size)
    data = []
    lab = []
    for i in range(num_batch):
        data.append(x_tr[i * size:i * size + size])
        lab.append(y_tr[i * size:i * size + size])
    return data, lab

'''
Xtr : [num_batch, batches, 784]
Ytr : [num_batch, batches, 784]
'''


#CNNモデル
def conv2d(x, W):#畳み込み層
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def convolutional_neural_network(x):
    weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 3])),
               'W_conv2': tf.Variable(tf.random_normal([5, 5, 3, 5])),
               'W_fc1': tf.Variable(tf.random_normal([7 * 7 * 5, 100])),
               'out': tf.Variable(tf.random_normal([100, labels_size]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([3])),
              'b_conv2': tf.Variable(tf.random_normal([5])),
              'b_fc1': tf.Variable(tf.random_normal([100])),
              'out': tf.Variable(tf.random_normal([labels_size]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    conv1 = tf.nn.sigmoid(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.sigmoid(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1, 7 * 7 * 5])
    fc = tf.nn.sigmoid(tf.matmul(fc, weights['W_fc1']) + biases['b_fc1'])

    output = tf.matmul(fc, weights['out']) + biases['out']

    return output

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

#ネットワークのデータ変数
training_data = tf.placeholder('float', [None, image_size * image_size * chan_num])
labels = tf.placeholder('float')

output = convolutional_neural_network(training_data)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=output))

correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



def update(l, a):
    global train_loss, train_acc
    train_loss = l
    train_acc = a


#Algorithm setting
for meth in algo:
    color = col[meth]

    #Adam
    if meth == 'Adam':
        '''
        alpha_k = [0.0001]
        mu = 0
        epoch = 25
        '''
        train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)
        timePlt = collections.deque(maxlen=iterations * epoch)
        errPlt = collections.deque(maxlen=iterations * epoch)
        alpha_k = collections.deque(maxlen=1)
        mu_val = collections.deque(maxlen=1)
        Xtr, Ytr = get_batches(Xtrain, Ytrain, batch_size)
    #oLMoQ_mb
    elif meth == 'oLMoQ_mb':
        m = 4
        sk_vec = collections.deque(maxlen=m)
        yk_vec = collections.deque(maxlen=m)
        gfk_vec = collections.deque(maxlen=1)
        gfk_vec_a = collections.deque(maxlen=1)
        alpha_k = collections.deque(maxlen=1)
        mu_val = collections.deque(maxlen=1)

        timePlt = collections.deque(maxlen=iterations * epoch)
        errPlt = collections.deque(maxlen=iterations * epoch)

        alpha_k.append(1)
        mu_val.append(0.85)
        vk_vec = collections.deque(maxlen=1)
        vk_vec.append(0)
        dirNorm = True

        gfk_vec = collections.deque(maxlen=2)
        xk_vec = collections.deque(maxlen=1)
        gfkp1 = collections.deque(maxlen=1)

        mini_grad = collections.deque(maxlen=3)
        using_grad = collections.deque(maxlen=3)

        Xtr, Ytr = get_batches(Xtrain, Ytrain, int(batch_size/3))

        train_step = tf.contrib.opt.ScipyOptimizerInterface(
            loss, method=meth.lower(),
            options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec, 'xk_vec': xk_vec, 'sk_vec': sk_vec,
                     'yk_vec': yk_vec,'gfk_vec_a': gfk_vec_a,
                     'timeplot': timePlt, 'err': errPlt, 'gfk_vec': gfk_vec,
                     'm': m, 'alpha_k': alpha_k, 'muk': mu_val, 'dirNorm': dirNorm})

        calc_full_grad = tf.contrib.opt.ScipyOptimizerInterface(loss, method='calc_full_grad',
                                                                options={'disp': True, 'mini_grad': mini_grad})

    else:
        m = 4
        epoch = 30
        sk_vec = collections.deque(maxlen=m)
        yk_vec = collections.deque(maxlen=m)
        gfk_vec = collections.deque(maxlen=1)
        gfk_vec_a = collections.deque(maxlen=1)
        alpha_k = collections.deque(maxlen=1)
        mu_val = collections.deque(maxlen=1)
        timePlt = collections.deque(maxlen=iterations * epoch)
        errPlt = collections.deque(maxlen=iterations * epoch)

        alpha_k.append(1)
        mu_val.append(0.8)
        vk_vec = collections.deque(maxlen=1)
        vk_vec.append(0)
        dirNorm = True
        Xtr, Ytr = get_batches(Xtrain, Ytrain, batch_size)

        if meth == 'oLNAQ' or meth == 'oLBFGS1'or meth == 'oLBFGS2':
            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method=meth.lower(),
                options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec,'gfk_vec': gfk_vec, 'sk_vec': sk_vec, 'yk_vec': yk_vec,
                         'timeplot':timePlt,'err':errPlt,
                         'm': m, 'alpha_k': alpha_k, 'muk': mu_val, 'dirNorm': dirNorm})

        elif meth == 'oLMoQ' or meth == 'oLMoQ_1':

            gfk_vec = collections.deque(maxlen=2)
            xk_vec = collections.deque(maxlen=1)

            train_step = tf.contrib.opt.ScipyOptimizerInterface(
                loss, method=meth.lower(),
                options={'maxiter': iterations, 'disp': False, 'vk_vec': vk_vec,'xk_vec': xk_vec, 'sk_vec': sk_vec, 'yk_vec': yk_vec,
                         'timeplot':timePlt,'err':errPlt,'gfk_vec':gfk_vec,
                         'm': m, 'alpha_k': alpha_k, 'muk': mu_val, 'dirNorm': dirNorm})

    # 学習開始
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    print("Initial Error of ", meth, " : ", sess.run(loss, feed_dict={training_data: Xtrain, labels: Ytrain}))

    step = 0
    alpha_k.append(1)

    eta0 = batch_size/(batch_size+2)

    test_acc_plot = []
    test_loss_plot = []
    train_acc_plot = []
    train_loss_plot = []


    for ep in range(1, epoch + 1):
        print("EPOCH ", ep, " : ##########")
        theta_k = 1

        if meth == 'oLMoQ_mb':
            #学習率の設定
            #if ep == 1 or ep%4==0:
            alpha_k.append(alpha_k[-1] * 0.5)
            #else:




            for i in range(iterations_mb-2):
                step += 1
                #alpha_k.append(1 / np.sqrt(step))
                #ミニバッチ取得
                data = []
                lab = []
                data = np.concatenate((Xtr[i], Xtr[i+1], Xtr[i+2]),axis=0)
                lab = np.concatenate((Ytr[i], Ytr[i + 1], Ytr[i + 2]), axis=0)

                feed_dict = {training_data: data, labels: lab}

                mu = mu_val[0]

                # alpha_k.append(0.1 * (epoch + 1 - ep))  # (0.01*(epoch+1-ep))#((np.sqrt(step)/(ep+step)))#*(batch_size+2)))#1 / (np.sqrt(step)))

                #oLMoQ によるパラメータ更新を実行
                res = train_step.minimize(sess, fetches=[loss, accuracy], loss_callback=update, feed_dict=feed_dict)

                #ミニミニバッチの勾配計算
                res = calc_full_grad.minimize(sess, fetches=[loss, accuracy], loss_callback=update,
                                                  feed_dict={training_data: Xtr[i], labels: Ytr[i]})
                gfkp1_1 = mini_grad[-1]

                res = calc_full_grad.minimize(sess, fetches=[loss, accuracy], loss_callback=update,
                                                  feed_dict={training_data: Xtr[i+1], labels: Ytr[i+1]})
                gfkp1_2 = mini_grad[-1]

                res = calc_full_grad.minimize(sess, fetches=[loss, accuracy], loss_callback=update,
                                                  feed_dict={training_data: Xtr[i+2], labels: Ytr[i+2]})
                gfkp1_3 = mini_grad[-1]

                #パラメータ更新後のミニバッチ勾配取得 : gfkp1
                full_g = np.zeros_like(mini_grad[0])
                full_g += mini_grad[0]
                full_g += mini_grad[1]
                full_g += mini_grad[2]
                gfkp1 = full_g / 3


                # sk yk の取得
                if i < 2:
                    sk = sk_vec[-1]
                    gfk = gfk_vec_a[-1]

                    yk = gfkp1 - gfk + sk

                else:
                    yk = gfkp1_1 - ((1 + mu) * using_grad[1] - mu * using_grad[0])

                using_grad.append(mini_grad[-2])
                using_grad.append(mini_grad[-1])

                gfk_vec.append(gfkp1)

                yk_vec.append(yk)

                #test loss, test acc 計算
                test_loss, test_acc = sess.run([loss, accuracy], feed_dict={training_data: Xtest, labels: Ytest})
                test_acc_plot.append(test_acc * 100)
                test_loss_plot.append(test_loss)
                train_acc_plot.append(train_acc * 100)
                train_loss_plot.append(train_loss)

                if i % 50 == 0:
                    print(
                        'Step {}; train loss {}; train accuracy {}; test loss {}; test accuracy {}; alpha {}; mu {}'.format(
                            i, train_loss, train_acc * 100, test_loss, test_acc * 100, alpha_k[0], mu_val[0]))

        else:
            #if ep == 1 or ep%4==0:
            if not meth == 'oLNAQ':
                alpha_k.append(alpha_k[-1] * 0.5)
            #else:


            for i in range(iterations):
                step += 1
                if meth == 'oLNAQ':
                    alpha_k.append(1 / np.sqrt(step))

                data = []
                lab = []
                data, lab = Xtr[i], Ytr[i]

                feed_dict = {training_data: data, labels: lab}

                if meth == 'Adam':
                    start = time.time()
                    _, train_loss, train_acc = sess.run([train_step, loss, accuracy], feed_dict=feed_dict)
                    end = time.time()
                    timePlt.append(end - start)

                else:
                    # alpha_k.append(1 / np.sqrt(step))  # (0.01*(epoch+1-ep))#((np.sqrt(step)/(ep+step)))#*(batch_size+2)))#1 / (np.sqrt(step)))
                    # alpha_k.append(0.1 * (epoch + 1 - ep))

                    #if step == 1:
                        # alpha_k.append(1 / np.sqrt(step))
                        # alpha_k.append(1 / (1+step))
                        #alpha_k.append(0.5)

                    if meth == 'oLBFGS1' or meth == 'oLBFGS2':
                        alpha_k.append(eta0 * (10 / (10 + step)))

                        mu_val.append(0)
                        mu = mu_val[-1]

                    if meth == 'oLNAQ':
                        #if step > 1: alpha_k.append(1 / np.sqrt(step))
                        mu = 0.85

                        mu_val.append(mu)
                        mu = mu_val[-1]

                    if meth == 'oLMoQ' or meth == 'oLMoQ_1':
                        #if step > 1: alpha_k.append(1 / np.sqrt(step))
                        # theta_kp1 = ((1e-5 - (theta_k * theta_k)) + np.sqrt(((1e-5 - (theta_k * theta_k)) * (1e-5 - (theta_k * theta_k))) + 4 * theta_k * theta_k)) / 2

                        mu = 0.85
                        mu_val.append(mu)
                        mu = mu_val[-1]
                        # alpha_k.append(0.1 * (epoch + 1 - ep))  # (0.01*(epoch+1-ep))#((np.sqrt(step)/(ep+step)))#*(batch_size+2)))#1 / (np.sqrt(step)))

                    res = train_step.minimize(sess, fetches=[loss, accuracy],
                                              loss_callback=update,
                                              feed_dict=feed_dict)

                test_loss, test_acc = sess.run([loss, accuracy], feed_dict={training_data: Xtest, labels: Ytest})
                test_acc_plot.append(test_acc * 100)
                test_loss_plot.append(test_loss)
                train_acc_plot.append(train_acc * 100)
                train_loss_plot.append(train_loss)


                if i % 50 == 0:
                    print(
                        'Step {}; train loss {}; train accuracy {}; test loss {}; test accuracy {}; alpha {}; mu {}'.format(
                            i, train_loss, train_acc * 100, test_loss, test_acc * 100, alpha_k[0], mu_val[0]))
                    '''print(
                        'Step {}; train loss {}; train accuracy {}; test loss {}; test accuracy {}'.format(
                            i, train_loss, train_acc * 100, test_loss, test_acc * 100))'''


    leg = algo

    plt.figure(5)
    plt.semilogy(train_loss_plot, color)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.legend(leg)
    plt.tight_layout()

    plt.figure(6)
    plt.semilogy(test_loss_plot, color)
    plt.xlabel('Iteration')
    plt.ylabel('Test Loss')
    plt.legend(leg)
    plt.tight_layout()

    plt.figure(7)
    plt.plot(train_acc_plot, color)
    plt.xlabel('Iteration')
    plt.ylabel('Train Accuracy')
    plt.legend(leg)
    plt.tight_layout()

    plt.figure(8)
    plt.plot(test_acc_plot, color)
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.legend(leg)
    plt.tight_layout()

    plt.figure(13)
    plt.semilogy(np.minimum.accumulate(train_loss_plot), color)
    plt.xlabel('Iteration')
    plt.ylabel('Training Loss')
    plt.legend(leg)
    plt.tight_layout()

    plt.figure(14)
    plt.semilogy(np.minimum.accumulate(test_loss_plot), color)
    plt.xlabel('Iteration')
    plt.ylabel('Test Loss')
    plt.legend(leg)
    plt.tight_layout()

    plt.figure(15)
    plt.plot(np.maximum.accumulate(train_acc_plot), color)
    plt.ylim((40,105))
    plt.xlabel('Iteration')
    plt.ylabel('Train Accuracy')
    plt.legend(leg)
    plt.tight_layout()

    plt.figure(16)
    plt.plot(np.maximum.accumulate(test_acc_plot), color)
    plt.ylim((40,100))
    plt.xlabel('Iteration')
    plt.ylabel('Test Accuracy')
    plt.legend(leg)
    plt.tight_layout()



    timePlt.clear()
    '''sk_vec.clear()
    yk_vec.clear()
    vk_vec.clear()'''

    sess.close()

print('seed: ', seed)
plt.show()



