from keras.optimizers import Adam
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.regularizers import l2
import keras.backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import model_from_json

import pandas as pd
import numpy as np
import cv2
import os
import scipy
from scipy import misc
from scipy import ndimage

def conv_block(ip, nb_filter, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional bottleneck block and dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        bottleneck: add bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added (optional bottleneck)
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)

    if bottleneck:
        inter_channel = nb_filter * 4 # Obtained from https://github.com/liuzhuang13/DenseNet/blob/master/densenet.lua

        x = Convolution2D(inter_channel, 1, 1, init='he_uniform', border_mode='same', bias=False,
                          W_regularizer=l2(weight_decay))(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
        x = Activation('relu')(x)

    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x

def transition_block(ip, nb_filter, compression=1.0, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional compression, dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(ip)
    x = Activation('relu')(x)
    x = Convolution2D(int(nb_filter * compression), 1, 1, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=False, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each conv_block is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of conv_block to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        bottleneck: bottleneck block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, bottleneck, dropout_rate, weight_decay)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis=concat_axis)
        nb_filter += growth_rate

    return x, nb_filter

def create_dense_net(img_dim, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=-1,
                     bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1E-4, verbose=True):
    ''' Build the create_dense_net model
    Args:
        nb_classes: number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay
        verbose: print the model type
    Returns: keras tensor with nb_layers of conv_block appended
    '''

    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"
    if reduction != 0.0:
        assert reduction <= 1.0 and reduction > 0.0, "reduction value must lie between 0.0 and 1.0"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    if bottleneck:
        nb_layers = int(nb_layers // 2)

    # compute initial nb_filter if -1, else accept users initial nb_filter
    if nb_filter <= 0:
        nb_filter = 2 * growth_rate

    # compute compression factor
    compression = 1.0 - reduction

    # Initial convolution
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D", bias=False,
                      W_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=bottleneck,
                                   dropout_rate=dropout_rate, weight_decay=weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, compression=compression, dropout_rate=dropout_rate,
                             weight_decay=weight_decay)
        nb_filter = int(nb_filter * compression)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, bottleneck=bottleneck,
                               dropout_rate=dropout_rate, weight_decay=weight_decay)

    x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('elu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='linear')(x)

    densenet = Model(input=model_input, output=x, name="create_dense_net")

    if verbose:
        if bottleneck and not reduction:
            print("Bottleneck DenseNet-B-%d-%d created." % (depth, growth_rate))
        elif not bottleneck and reduction > 0.0:
            print("DenseNet-C-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        elif bottleneck and reduction > 0.0:
            print("Bottleneck DenseNet-BC-%d-%d with %0.1f compression created." % (depth, growth_rate, compression))
        else:
            print("DenseNet-%d-%d created." % (depth, growth_rate))


    return densenet

def turn_feature_layers_non_trainable(dense_net):
    '''Turn feature layers to non traniable for transffering
    Args:
        dense_net: A keras model is created by create_dense_net()
    Return:
        Another keras model, only the last layer (regression) is trainable
    '''
    #trun all layers into non trainable except the last lear
    last_layer = dense_net.layers[-1]
    for l in dense_net.layers:
        if l == last_layer:
            continue
        else:
            l.trainable = False
    return dense_net

def train(network, model_fn, x, y, from_scratch=False, epoch=1, batch=64, early_stop = -1, val_split_ratio = 0.1):
    '''A wrapper of keras fit() function for regression output (with mse loss)
    Args:
        network: A keras model
        model_fn: The file name of weight (.h5) and model architecture description file (.json).
        x: Input data
        y: Label data
        replace: Is from scratch or from exists weight file?
        epoch: The number of training iteration
        batch: The number of sample in each training batch
        early_stop: The number of epoch patience if the validation performance is better. -1 means no early stop.
        val_split_ratio: The ratio of train/val splition (shuffle)
    Return: The result of keras fit() return value
    '''
    if os.path.exists(model_fn) and from_scratch == False:
        #continous training
        network.load_weights(model_fn)

    #model.summary()
    optimizer = Adam(lr=1e-3)
    network.compile(loss='mse', optimizer=optimizer, metrics=None)
    network.summary()

    cb = [ModelCheckpoint(model_fn, monitor="val_loss", save_best_only=True, save_weights_only=True)]

    if early_stop != -1:
        cb.append(EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop, verbose=0, mode='auto'))

    result = network.fit(x, y, batch_size=batch, nb_epoch=epoch,
                   validation_split=val_split_ratio,
                   shuffle = True,
                   callbacks=cb,
                   verbose=2)

    #print(model.history)
    result.model.save_weights(model_fn)
    return result

def predict(x):
    '''The wrapper of keras predict function
    Args:
        x: A keras model is created by create_dense_net()
    Return:
        Another keras model, only the last layer (regression) is trainable
    '''
    model_fn = 'model.json'

    with open(model_fn, 'r') as jfile:
        network = model_from_json(jfile.read())

    network.compile("adam", "mse")
    weights_file = model_fn.replace('json', 'h5')
    network.load_weights(weights_file)

    pred = network.predict(x)
    return pred

def parse_driving_log_and_load_image(csv_fn, image_shape):
    '''Parse the csv file to get relative date (steer angle, throttle,..) and read image (BGR)
    Args:
        csv_fn: String of the filename of csv
        image_shape: Output image size.
    Return:
        3 pairs of camera frame and steer angle: left , center and right camera
    '''
    driving_log = pd.read_csv(csv_fn, delimiter=',', header=None, skiprows=1)
    img_dir = os.path.split(csv_fn)[0]
    images_center = []
    images_left = []
    images_right = []
    attributes_center = []
    attributes_left = []
    attributes_right = []

    for i in range(len(driving_log)):
        img_path_center = os.path.join(os.path.join(os.path.join(img_dir, 'IMG')), os.path.split(driving_log[0][i])[1])
        img_path_left = os.path.join(os.path.join(os.path.join(img_dir, 'IMG')), os.path.split(driving_log[1][i])[1])
        img_path_right = os.path.join(os.path.join(os.path.join(img_dir, 'IMG')), os.path.split(driving_log[2][i])[1])
        #print(img_path)
        angle = driving_log[3][i]
        throttle = driving_log[4][i]
        bark = driving_log[5][i]
        speed = driving_log[6][i]

        #skip the frame that speed == 0
        if speed < 0e-2:
            continue

        #note:
        #cv2.imread => [h, w, c] (BGR)
        #scipy.ndimage.imread => [h, w, c] (RGB)
        img_center = cv2.imread(img_path_center)
        img_left = cv2.imread(img_path_left)
        img_right = cv2.imread(img_path_right)

        #img_center = scipy.ndimage.imread(img_path_center)
        #img_left = scipy.ndimage.imread(img_path_left)
        #img_right = scipy.ndimage.imread(img_path_right)

        if img_center is not None:
            img_resized = cv2.resize(img_center, image_shape)
            images_center.append(img_resized)
            attributes_center.append(angle)
        else:
            print("Image file can't be loaded:", img_path_center)

        if img_left is not None:
            img_resized = cv2.resize(img_left, image_shape)
            images_left.append(img_resized)
            attributes_left.append(angle)
        else:
            print("Image file can't be loaded:", img_path_left)

        if img_right is not None:
            img_resized = cv2.resize(img_right, image_shape)
            images_right.append(img_resized)
            attributes_right.append(angle)
        else:
            print("Image file can't be loaded:", img_path_right)

    a = (np.array(images_center), np.array(attributes_center))
    b = (np.array(images_left), np.array(attributes_left))
    c = (np.array(images_right), np.array(attributes_right))
    return a, b, c

def load_data(root_dir, image_shape):
    '''Load data recursively
    Args:
        root_dir: The root folder of training data
        image_shape: Output image size.
    Return:
        3 pairs of camera frame and steer angle: left , center and right camera
    '''
    data_x_c = []
    data_y_c = []
    data_x_l = []
    data_y_l = []
    data_x_r = []
    data_y_r = []

    for root, dirs, files in os.walk(root_dir):
        if "driving_log.csv" in files:
            csv_path = os.path.join(root, "driving_log.csv")
            #(x_c, y_c), (x_l, y_l), (x_r, y_r) = parse_driving_log_and_load_image(csv_path, image_shape)
            a,b,c = parse_driving_log_and_load_image(csv_path, image_shape)
            (x_c, y_c), (x_l, y_l), (x_r, y_r) = a,b,c
            data_x_c.append(x_c)
            data_y_c.append(y_c)
            data_x_l.append(x_l)
            data_y_l.append(y_l)
            data_x_r.append(x_r)
            data_y_r.append(y_r)

    a = (np.concatenate(data_x_c), np.concatenate(data_y_c))
    b = (np.concatenate(data_x_l), np.concatenate(data_y_l))
    c = (np.concatenate(data_x_r), np.concatenate(data_y_r))
    return a, b, c

#BGR => BGRG
def preprocess_insert_gray_channel(data):
    '''Converting original image data into gray then append to original frame
    Args:
        data: The training image
    Return:
        training image with BGRG
    '''
    n_sample = data.shape[0]
    new_images = []
    for index in range(n_sample):
        img = data[index]
        #plt.imshow(final)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        b,g,r = cv2.split(img)
        bgrg_img = cv2.merge((b,g,r,gray))
        new_images.append(bgrg_img)
    return np.array(new_images)

#param:
#x_l, x_c, x_r: left camera frames, center camera frames, right camera frames
#y_c: center camera angle
def augment_data_by_side_camera(x_l, x_c, x_r, y_c):
    '''Combine 3 camera data
    Args:
        x_l: Left camera frame
        x_c: Center camera frame
        x_r: Right camera frame
        y_c: Steer angle
    Return:
        Training data with 3 camera
    '''
    cal_factor = np.std(y_c)
    cal_l = cal_factor * (1-abs(y_c))
    cal_r = -cal_factor * (1-abs(y_c))
    y_l = y_c + cal_l
    y_r = y_c + cal_r

    x_aug = np.vstack((x_l, x_c, x_r))
    y_aug = np.hstack((y_l, y_c, y_r))

    return x_aug, y_aug

def augment_data_by_flip(images, angles):
    '''Apply horizontal flip
    Args:
        images: camera frame images
        angles: steer angles
    Return:
        Orignal data with horizontal data
    '''

    #flip data augmentation
    flip_images = []
    flip_angle = []
    for i, img in enumerate(images):
        flip_images.append(cv2.flip(img, 1))
        flip_angle.append(angles[i]*-1)

    x_aug = np.vstack((images, np.array(flip_images)))
    y_aug = np.hstack((angles, np.array(flip_angle)))
    return x_aug, y_aug

if __name__ == "__main__":
    #environment setup
    root_dir = "../../../../data/Behavior_Cloning-P2/normal"
    model_dn_fn = "./model-dn"
    model_final_fn = "./model"

    #configuration
    height = 80
    width = 80
    channel = 3
    image_shape = (height, width)

    #load data
    (x_train_c, y_train_c), (x_train_l, y_train_l), (x_train_r, y_train_r) = load_data(root_dir, image_shape)
    #data augmentation by combining 3 camera frames
    x_train, y_train = augment_data_by_side_camera(x_train_l, x_train_c, x_train_r, y_train_c)
    #data preprocess: append gray channel
    x_train_processed = preprocess_insert_gray_channel(x_train)
    #data augmentation by combining flip
    x_train_augmented, y_train_augmented = augment_data_by_flip(x_train_processed, y_train)
    print("x_train_augmented shape:", x_train_augmented.shape)
    print("y_train_augmented shape:", y_train_augmented.shape)
    #train model #1
    dense_net = create_dense_net((height, width, channel+1), 3*4+4, 3, 8, -1, True, 0.5, 0)
    json_string = dense_net.to_json()

    with open(model_dn_fn + ".json", "w") as json_file:
        json_file.write(json_string)
        from_scratch = False
        result = train(dense_net, model_dn_fn + ".h5", x_train_augmented, y_train_augmented, from_scratch, epoch=30)

    #train model #2, only use center camera
    #data preprocess: append gray channel
    x_train_processed = preprocess_insert_gray_channel(x_train_c)
    #data augmentation by combining flip
    x_train_augmented, y_train_augmented = augment_data_by_flip(x_train_processed, y_train_c)
    print("x_train_augmented shape:", x_train_augmented.shape)
    print("y_train_augmented shape:", y_train_augmented.shape)
    #phase 2: Transferring learning (only on center camera)
    with open(model_dn_fn + ".json", 'r') as jfile:
        dense_net_final = model_from_json(jfile.read())
        #load model from file
        dense_net_final.load_weights(model_dn_fn+".h5")
        #dn.summary()
        dense_net_final = turn_feature_layers_non_trainable(dense_net_final)
        with open(model_final_fn + ".json", "w") as json_file:
            json_string = dense_net_final.to_json()
            json_file.write(json_string)

        from_scratch = True
        result_final = train(dense_net_final, model_final_fn + ".h5", x_train_augmented, y_train_augmented, from_scratch, 30, 128, 3)

    import gc; gc.collect()
