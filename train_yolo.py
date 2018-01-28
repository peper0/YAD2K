"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
import argparse
import logging

import os
from datetime import datetime
from random import shuffle
from typing import  Optional, Generator, Tuple, NewType, Sequence, List, Union

import numpy as np
import PIL
import tensorflow as tf
from keras import backend as K, optimizers
from keras.layers import Input, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LambdaCallback
from scipy.ndimage.io import imread

from yad2k.models.keras_yolo import (preprocess_true_boxes,
                                     yolo_eval, yolo_head, YoloLossLayer, Anchors, IBoxes)
from yad2k.utils.draw_boxes import draw_boxes

#: Image packed in the numpy array; shape is [height, width, colors]; values in range 0..255
Image = NewType('Image', np.array)

#: indexing: [box_index, [class_index, px_min, py_min, px_max, py_max]]
Boxes = NewType('Boxes', np.array)

DataGenerator = Generator[Tuple[Image, Boxes], None, None]

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
#    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    help="path to directory where files like xxx.jpg and xxx.txt are stored")

argparser.add_argument(
    '-l',
    '--valid_data_path',
    help="path to directory where validation data is stored (see also --data-path)")

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors filet')

argparser.add_argument(
    '-c',
    '--classes_path',
    required=True,
    help='path to classes file, defaults to pascal_classes.txt')

argparser.add_argument(
    '-m',
    '--model_path',
    help='path to h5 model file containing body of a YOLO_v2 model')

argparser.add_argument(
    '-r',
    '--reset_final',
    action='store_true',
    help='resets final layer; necessary if we want to train for different classes than the loaded model was')


# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


IMAGE_SIZE = (608, 608)
OUTPUT_IMG_SHAPE = (19, 19)


def data_generator(path, class_names: Sequence[str], infinite = True) -> DataGenerator:
    while True:
        filelist = list(os.listdir(path))
        shuffle(filelist)
        for filename in filelist:
            filename = os.path.join(path, filename)
            if filename.endswith(".jpg"):
                try:
                    label_filename = filename.replace(".jpg", ".txt")
                    image = imread(filename)
                    boxes = []
                    with open(label_filename, 'r') as f:
                        for line in f.readlines():
                            class_index, center_x, center_y, width, height = map(float, tuple(line.split()))
                            xmin = (center_x - width/2)*image.shape[1]
                            ymin = (center_y - height/2)*image.shape[0]
                            xmax = (center_x + width/2)*image.shape[1]
                            ymax = (center_y + height/2)*image.shape[0]
                            boxes.append((class_index, xmin, ymin, xmax, ymax))
                        if len(boxes) == 0:
                            raise Exception("no true boxes!")
                    yield image, np.array(boxes)
                except Exception:
                    logging.exception("exception during reading image or labels for '{}'; ignoring".format(filename))

        if not infinite:
            return


def _main(args):
    data_path = os.path.expanduser(args.data_path)
    valid_data_path = os.path.expanduser(args.valid_data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)
    model_path = os.path.expanduser(args.model_path)

    class_names = load_classes(classes_path)
    anchors = load_anchors(anchors_path)
    model_body = prepare_model(model_path, anchors, class_names, args.reset_final)
    trainable_model = compile_model(model_body, anchors, class_names)

    train(
        trainable_model,
        model_body,
        anchors,
        data_generator(data_path, class_names, infinite=True),
        data_generator(valid_data_path, class_names, infinite=False)
    )


def load_classes(classes_path) -> List[str]:
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def load_anchors(anchors_path) -> Anchors:
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS


def arraize_data(images: Sequence[Image], boxes: Optional[Sequence[Boxes]]=None) \
        -> Union[Tuple[np.ndarray, IBoxes], np.ndarray]:
    '''processes the data
    :returns: (np.array of resized images with normalized channels, shape [batch_size, height, width]),
              (np.array of shape [batch_size, max(num_boxes), 5], each box in the form
              [ix_center, iy_center, iw, ih, class])

    '''
    images = [PIL.Image.fromarray(i) for i in images]
    def orig_size(image):
        res = np.array([image.width, image.height])
        return np.expand_dims(res, axis=0)

    # Image preprocessing.
    #FIXME: input size?
    processed_images = [i.resize(IMAGE_SIZE, PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        #boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_pxy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_pwh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_ixy = [boxxy / orig_size(img) for boxxy, img in zip(boxes_pxy, images)]
        boxes_iwh = [boxwh / orig_size(img) for boxwh, img in zip(boxes_pwh, images)]
        boxes = [np.concatenate((boxes_ixy[i], boxes_iwh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

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


def get_detector_mask(boxes: IBoxes, anchors: Anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0] * len(boxes)
    matching_true_boxes = [0] * len(boxes)
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, IMAGE_SIZE)

    return np.array(detectors_mask), np.array(matching_true_boxes)


def freeze_body(model_body, freeze):
    topless_yolo = Model(model_body.input, model_body.layers[-2].output)
    for layer in topless_yolo.layers:
        layer.trainable = not freeze


def final_filters_count(num_anchors, num_classes):
    return num_anchors * (5 + num_classes)


def prepare_model(model_path: str, anchors: Anchors, class_names: Sequence[str], reset_final: bool) -> (Model, Model):
    '''
    returns the body of the model
    '''

    loaded_model = load_model(model_path)

    final_filters = final_filters_count(len(anchors), len(class_names))

    if reset_final:
        topless_yolo = Model(loaded_model.input, loaded_model.layers[-2].output)
        #print("====topless:")
        #topless_yolo.summary()

        #
        # if load_pretrained:
        #     # Save topless yolo:
        #     topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        #     topless_yolo.load_weights(topless_yolo_path)

        final_layer = Conv2D(final_filters, (1, 1), activation='linear', name="final_conv")(topless_yolo.output)

        model_body = Model(topless_yolo.input, final_layer)
        #print("====body:")
        #model_body.summary()
    else:
        model_body = loaded_model

    assert model_body.layers[-1].output_shape[3] == final_filters, \
        "loaded model was trained for a different number of classes than found in the classes file; use --reset_final" \
        " option to reset the final layer"
    return model_body


def metric_from_index(index, name):
    def metric(y_true, y_pred):
        return K.mean(y_pred[index])

    m = metric
    m.__name__ = name
    return m


def compile_model(model_body: Model, anchors: Anchors, class_names: Sequence[str]) -> Model:
    """
    Prepares model takes yolo body model and returns the compiled model that can be trained. This model has additional
    inputs that are used in the training:
    1. true_boxes in the form of array [box_index, [x_center, y_center, width, height, class]]
    2. detectors_mask with index [cell_y, cell_x, anchor_index, 1] - 1 if there should be any detection and
    3. matching_boxes [cell_y, cell_x, anchor_index, [dy, dx, th, tw, class]]
    """
    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # Create model input layers.
        num_anchors = len(anchors)
        detectors_mask_shape = OUTPUT_IMG_SHAPE + (num_anchors, 1)
        matching_boxes_shape = OUTPUT_IMG_SHAPE + (num_anchors, 5)
        # Additional inputs for the loss function
        boxes_input = Input(shape=(None, 5), name="boxes_input")
        detectors_mask_input = Input(shape=detectors_mask_shape, name="detectors_mask_input")
        matching_boxes_input = Input(shape=matching_boxes_shape, name="matching_boxes_input")

        # Loss function must be represented as a layer since in keras the truth during training has the same shape as
        # the network output.
        model_loss = YoloLossLayer(anchors, num_classes=len(class_names), print_loss=False, name="yolo_loss")([
            model_body.output,
            boxes_input,
            detectors_mask_input,
            matching_boxes_input
        ])

    trainable_model = Model(
        [model_body.input, boxes_input, detectors_mask_input, matching_boxes_input], model_loss)

    trainable_model.compile(
        # parameters copied from yolo.cfg
        optimizer=optimizers.Adam(lr=0.0001),
        #optimizer=optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005),
        loss=YoloLossLayer.loss_function,
        metrics=[metric_from_index(1, 'cxy_rmse'),
                 metric_from_index(2, 'awh_rmse'),
                 metric_from_index(3, 'good_iou_avg'),
                 metric_from_index(4, 'good_conf_avg'),
                 metric_from_index(5, 'coords_loss'),
                 metric_from_index(6, 'confidence_loss'),
                 metric_from_index(7, 'class_loss')]
    )

    return trainable_model


def train(trainable_model: Model,
          model_body: Model,
          anchors: Anchors,
          data_generator: DataGenerator,
          valid_data_generator: DataGenerator
          ):
    '''
    Retrain/fine-tune the model. Logs training with tensorboard. Saves training weights in current directory.

    best weights according to val_loss is saved as trained_stage_3_best.h5

    '''

    tensorboard = TensorBoard(log_dir='.',
                              #histogram_freq=1,
                              write_graph=True,
                              #write_grads=True,
                              #write_images=True,
                              #embeddings_freq=1,
                              #embeddings_layer_names=['conv2d_21'],
                              embeddings_metadata=dict(conv2d_21="conv2d_21")
                              )
    checkpoint = ModelCheckpoint("trained_best.h5", monitor='loss',
                                 save_weights_only=False, save_best_only=True)
    early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=15, verbose=1, mode='auto')

    def save(epoch, logs):
        print("saving model...")
        if epoch % 50 == 1:
            model_body.save('trained_epoch_{}.h5'.format(datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
        model_body.save('trained_last.h5')

    my_callback = LambdaCallback(on_epoch_end=save)

    def batch_generator(user_data_generator: DataGenerator, batch_size=1):
        images = []
        boxess = []

        def push_data(images, boxess: Sequence[Boxes]):
            assert len(images) > 0
            logging.info("gathered batch, processing it")
            images_array, boxes_array = arraize_data(images, boxess)
            detectors_mask, matching_true_boxes = get_detector_mask(boxes_array, anchors)

            yield ([images_array, boxes_array, detectors_mask, matching_true_boxes],  # network input
                   np.zeros(len(images_array)))  # network expected output

        for image, boxes in user_data_generator:
            images.append(image)
            boxess.append(boxes)
            if len(images) >= batch_size:
                yield from push_data(images, boxess)
                images = []
                boxess = []

        if len(images) > 0:
            yield from push_data(images, boxess)

    validation_data = next(iter(batch_generator(valid_data_generator)))

    steps_per_epoch = 200
    common_fit_args = dict(steps_per_epoch=steps_per_epoch, validation_data=validation_data)

    ####### STAGE1 - train only the last layer
    #freeze_body(model_body, True)
    if False:
        trainable_model.fit_generator(batch_generator(data_generator),
                                      epochs=5,
                                      callbacks=[tensorboard],
                                      **common_fit_args)

        model_body.save('trained_stage_1.h5')

        ####### STAGE2
        #freeze_body(model_body, False)
        trainable_model.fit_generator(batch_generator(data_generator),
                                      epochs=10,
                                      callbacks=[tensorboard],
                                      **common_fit_args)

        model_body.save('trained_stage_2.h5')

    ####### STAGE3 - like stage2?
    trainable_model.fit_generator(batch_generator(data_generator),
                                  epochs=30000,
                                  callbacks=[tensorboard, checkpoint, my_callback],
                                  **common_fit_args)

    model_body.save('trained_stage_3.h5')

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
