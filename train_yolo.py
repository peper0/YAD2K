"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse

import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head, yolo_loss)
from yad2k.utils.draw_boxes import draw_boxes

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
#    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    help="path to directory where files like xxx.jpg and xxx.txt are stored",
    default=os.path.join('..', 'DATA', 'underwater_data.npz'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'underwater_classes.txt'))

argparser.add_argument(
    '-m'
    '--model_path',
    help='path to h5 model file containing body of a YOLO_v2 model')

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    model_path = os.path.expanduser(args.model_path)

    class_names = load_classes(classes_path)
    anchors = load_anchors(anchors_path)
    model_body, trainable_model = prepare_model(anchors, class_names, model_path)

    #FIXME
    data = np.load(data_path) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'

    #FIXME
    image_data, boxes = process_data(data['images'], data['boxes'])

    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    train(
        trainable_model,
        model_body
        class_names,
        anchors,
        FIXME
    )

    draw(model_body,
        class_names,
        anchors,
        image_data,
        image_set='val', # assumes training/validation split is 0.9
        weights_name='trained_stage_3_best.h5',
        save_all=False)


def load_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def load_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


def process_data(images: Iterable[np.array], boxes: Optional[Iterable[np.array]]=None):
    '''processes the data
    :param images:
    :param boxes: iterable of np.array objects, each of shape [num_boxes, 5]
                  with coordinates [index, (class, x_min, y_min, x_max, y_max)], using pixel coordinates

    :returns: (np.array of resized images with normalized channels, shape (batch_size,)),
              (np.array of shape (batch_size, max(num_boxes), 5), each box in the form
              (y_center, x_center, height, width), everything as a fraction of proper image dimension)

    '''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    #FIXME: input size?
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)


def freeze_body(model_body, freeze):
    topless_yolo = Model(model_body.input, model_body.layers[-2].output)
    for layer in topless_yolo.layers:
        layer.trainable = not freeze


def prepare_model(anchors, class_names, model_path):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    trainable_model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    yolo_model = load_model(model_path)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model2 = yolo_body(image_input, len(anchors), len(class_names))
    print(yolo_model2.summary())
    print(yolo_model.summary())
    #from keras.utils.vis_utils import plot_model as plot
    #plot(yolo_model, to_file='{}.png'.format("yolo_body"), show_shapes=True)

    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)
    #
    # if load_pretrained:
    #     # Save topless yolo:
    #     topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
    #     topless_yolo.load_weights(topless_yolo_path)

    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    trainable_model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    trainable_model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    return model_body, trainable_model


def train(trainable_model, model_body, class_names, anchors, data_generator):
    '''
    Retrain/fine-tune the model. Logs training with tensorboard. Saves training weights in current directory.

    best weights according to val_loss is saved as trained_stage_3_best.h5
    :param data_generator: a generator of tuples (images, boxes), where images is a list of np.array's and boxes is
                           a list of np.array's with (class_name, x_min, y_min, x_max, y_max), using pixel coordinates
    '''

    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_stage_3_best.h5", monitor='val_loss',
                                 save_weights_only=False, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')
    steps_per_epoch = 1000  # fixme
    common_fit_args = dict(steps_per_epoch=steps_per_epoch)

    def loss_input_generator(user_data_generator):
        for images, boxes in user_data_generator:
            image_data, boxes = process_data(images, boxes)
            detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)
            yield (image_data, boxes, detectors_mask, matching_true_boxes), np.zeros(len(image_data)),

    ####### STAGE1 - train only the last layer
    freeze_body(model_body, True)
    trainable_model.fit_generator(loss_input_generator(data_generator),
                                  epochs=5,
                                  callbacks=[logging],
                                  **common_fit_args)

    trainable_model.save_weights('trained_stage_1.h5')

    ####### STAGE2 - train only the last layer
    freeze_body(model_body, False)
    trainable_model.fit_generator(loss_input_generator(data_generator),
                                  epochs=30,
                                  callbacks=[logging],
                                  **common_fit_args)

    trainable_model.save_weights('trained_stage_2.h5')

    ####### STAGE3 - like stage2?
    trainable_model.fit_generator(loss_input_generator(data_generator),
                                  epochs=30,
                                  callbacks=[logging, checkpoint, early_stopping],
                                  **common_fit_args)

    trainable_model.save_weights('trained_stage_3.h5')

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5', out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0)

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })
        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()



if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
