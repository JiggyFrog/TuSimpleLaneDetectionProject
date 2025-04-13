import datetime
import glob
import json
import re

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# sets
sets = ["0531", "0601", "313"]

# Define the log directory
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Create the TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# HYPERS
batch_size = 200
training_batch_size = 5
epochs = 4
load = False
training_set = sets[0]
threads = 8


# HYPERS


# @tf.keras.utils.register_keras_serializable("package")
class CBAM(tf.keras.layers.Layer):
    """CBAM attention module to improve model performance"""

    def __init__(self, reduction_ratio=16, trainable=False, dtype=False, name="CBAM"):
        super(CBAM, self).__init__()
        self.reduction_ratio = reduction_ratio
        print(trainable, dtype)

    def build(self, input_shape):
        _, height, width, channels = input_shape

        # Channel Attention Module
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.max_pool = tf.keras.layers.GlobalMaxPooling2D()

        self.fc1 = tf.keras.layers.Dense(channels // self.reduction_ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')

        # Spatial Attention Module
        self.conv = tf.keras.layers.Conv2D(1, kernel_size=7, padding='same', activation='sigmoid')

    def call(self, inputs):
        # Channel Attention
        avg_pooled = self.avg_pool(inputs)
        max_pooled = self.max_pool(inputs)

        avg_out = self.fc2(self.fc1(avg_pooled))
        max_out = self.fc2(self.fc1(max_pooled))

        channel_out = tf.keras.layers.Multiply()([inputs, avg_out + max_out])

        # Spatial Attention
        avg_pool_spatial = tf.reduce_mean(channel_out, axis=-1, keepdims=True)
        max_pool_spatial = tf.reduce_max(channel_out, axis=-1, keepdims=True)

        spatial_out = tf.concat([avg_pool_spatial, max_pool_spatial], axis=-1)
        spatial_out = self.conv(spatial_out)

        return tf.keras.layers.Multiply()([channel_out, spatial_out])


def se_block(input_tensor, ratio=8):
    """SE_Block to improve model performance"""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)
    return layers.Multiply()([input_tensor, se])


def iou_loss(y_true, y_pred):
    """DEPRECATED, Intersection over Union loss"""
    # Ensure predictions are clipped to valid range
    y_pred = tf.clip_by_value(y_pred, 0, 1)

    # Calculate intersection and union areas
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred), axis=[1, 2])
    union = tf.reduce_sum(tf.maximum(y_true, y_pred), axis=[1, 2])

    # Calculate IoU
    iou = intersection / (union + 1e-7)  # Add epsilon to avoid division by zero

    # Return IoU loss (1 - IoU)
    return 1 - tf.reduce_mean(iou)


tf.config.threading.set_intra_op_parallelism_threads(threads)


def build_model():
    """Builds a tensorflow Model"""
    input_layer = layers.Input(shape=(640, 360, 3))

    # Convolutional Block 1
    x = layers.Conv2D(64, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform')(
        input_layer)
    x = layers.BatchNormalization()(x)

    # Convolutional Block 2
    x = layers.Conv2D(128, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = CBAM()(x)  # Add CBAM block
    # Convolutional Block 3
    x = layers.Conv2D(256, (3, 3), strides=2, padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense Layers
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # Output Layer
    x = layers.Dense(4 * 48, activation='tanh')(x)
    output_layer = layers.Reshape((4, 48))(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()
    return model


# Get our directories that we will be using
file_paths = glob.glob("TUSimple/train_set/clips/0313-1/**", recursive=True)
file_paths.extend(glob.glob("TUSimple/train_set/clips/0313-2/**", recursive=True))
tmp = []
cs = len(r"TUSimple\\train_set\\clips\\0313-1")

for x in file_paths:  # Find all of the images that will have matches
    if x[-6:] == '20.png' or x[-6:] == '20.jpg':
        tmp.append(x)

file_paths = tmp
del tmp
print(len(file_paths))


def load_json(p):
    """Load Json Path"""
    tmp = []
    with open(p, 'r') as f:
        for line in f.readlines():
            tmp.append(json.loads(line))
    return tmp


# TUSimple/train_set/clips/0313-2\\7920\\20.png


def get_v_from_path(p):
    """Takes an image path and convert it to the Images signature to be linked with its dataset partner"""
    spl = re.split(r"[/\\]", p)
    print(spl[4] + p[30])
    return spl[4] + p[30]


def get_v_from_raw_file(r: str):
    """Gets the file encoding from what the dataset  to link the Dataset V with the image V
    ex: if Dataset V==Image V then they are supposed to go together, I had to do this because the dataset was a bit
    of a mess
    """
    return r.split("/")[2] + r[11]


def normalize_with_special(arr, special_value):
    """Normalizes an array, handling a special value."""

    # Create a mask for the special value
    mask = arr != special_value
    mask2 = arr == special_value
    # Normalize the array, excluding the special value
    arr[mask] = arr[mask] / 1280
    arr[mask2] = -1
    return arr


json_y_labels = load_json("TUSimple/train_set/label_data_0313.json")
y_ordered = []
paths_ordered = []
for path in file_paths:
    path_v = get_v_from_path(path)
    for r in json_y_labels:
        if path_v == get_v_from_raw_file(r["raw_file"]):
            y_ordered.append(normalize_with_special(np.array(r["lanes"], np.float32), -2))
            paths_ordered.append(path)
            break
ds_length = len(y_ordered)
image = tf.io.read_file(file_paths[5])
image = tf.image.decode_jpeg(image, channels=3)
image = tf.image.resize(image, [640, 360])
image = tf.cast(image, tf.float32) / 255.0  # Normalize
print(image)


def preprocess_images(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [640, 360])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label


training = tf.data.Dataset.from_tensor_slices((paths_ordered, y_ordered))
x_training = training.map(preprocess_images)

backup_checkpoint = tf.keras.callbacks.BackupAndRestore(
    # The path where your backups will be saved. Make sure the
    # directory exists prior to invoking `fit`.
    "./training_backup",
    # How often you wish to save a checkpoint. Providing "epoch"
    # saves every epoch, providing integer n will save every n steps.
    save_freq="epoch",
    # Deletes the last checkpoint when saving a new one.
    delete_checkpoint=True,
)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="best_model_checkpoint.keras",  # Path to save the checkpoints
    save_weights_only=False,  # Save only model weights
    save_best_only=True,  # Save only the best model based on validation performance
    monitor="loss",  # Metric to monitor for improvement
    mode='min'
)


def combined_loss(y_true, y_pred):
    iou = iou_loss(y_true, y_pred)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))  # Mean Absolute Error
    return 0.2 * iou + mae  # Weight MAE contribution


if not load:
    model = build_model()
else:
    model = load_model('best_model_checkpoint.keras', custom_objects={"combined_loss": combined_loss, "CBAM": CBAM})
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.002), loss='mae', metrics=['mae'])
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                              patience=5, min_lr=0.0001)
epochs_trained = 0
model.build(input_shape=(640, 360, 3))
print(y_ordered)
x_training = x_training.shuffle(400)
x_training = x_training.batch(5)
model.fit(
    x_training,
    epochs=epochs,
    callbacks=[checkpoint_callback, reduce_lr, backup_checkpoint, tensorboard_callback],
)
model.save("model.keras")
