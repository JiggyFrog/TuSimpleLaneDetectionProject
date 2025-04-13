import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import glob
import cv2

imgDir = "./TUSimple/test_set/clips/0530/1492626158152981904_0/10.jpg" #Image directory
height_samples = [240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430,
                  440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610, 620, 630,
                  640, 650, 660, 670, 680, 690, 700, 710]


class CBAM(tf.keras.layers.Layer):
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


def iou_loss(y_true, y_pred):
    """DEPRECATED Intersection over Union Loss function"""
    # Calculate intersection area
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred), axis=[1, 2])

    # Calculate union area
    union = tf.reduce_sum(tf.maximum(y_true, y_pred), axis=[1, 2])

    # Calculate IoU
    iou = intersection / union

    # Return IoU loss (1 - IoU)
    return 1 - iou


def preprocess_images(path):
    """Pre processes an image into something the model can read"""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [640, 360])
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image


def combined_loss(y_true, y_pred):
    iou = iou_loss(y_true, y_pred)
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))  # Mean Absolute Error
    return 0.5 * iou + mae  # Weight MAE contribution

# Loads ts model
model = load_model("best_model_checkpoint.keras", custom_objects={"combined_loss": combined_loss, "CBAM": CBAM})
img = cv2.imread(imgDir)


def preprocess_video(path, output_dir, cap_every_nth=10):
    cap = cv2.VideoCapture(path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print("Error opening video file")
        return

    import os
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    while True:

        ret, frame = cap.read()

        if not ret:
            break
        if not frame_count % cap_every_nth == 0.0:
            frame_count += 1
            continue
        output_path = os.path.join(output_dir, f"frame_{frame_count: 04d}.jpg")
        cv2.imwrite(output_path, frame)

        frame_count += 1

    cap.release()


def draw_lane(lane, img, c):
    for index in range(len(height_samples)):
        cv2.circle(img, (round(lane[index]), height_samples[index]), 1, c, 5)


def make_video(path, tmp, output):
    #preprocess_video(path, tmp)
    all = glob.glob(f'{tmp}*')
    print(all)
    ds = tf.data.Dataset.from_tensor_slices(all)
    ds = ds.map(preprocess_images)
    batches = ds.batch(batch_size=24)
    video_lanes = []
    for batch in batches:
        video_lanes.extend(model.predict(batch))
    cv2_fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    size = [1280, 720]
    print(video_lanes)
    video = cv2.VideoWriter(output, cv2_fourcc, 3, size)
    for frame, lanes, path in zip(range(len(video_lanes)), video_lanes, all):
        img = cv2.imread(path)
        for lane in lanes:
            draw_lane(lane * 1280, img, (255, 255, 255))
        video.write(img)

    exit(4)


def show_image():
    lanes = model.predict(np.array([preprocess_images(imgDir)]))[0]
    print(lanes)
    draw_lane(lanes[0] * 1280, img, (255, 0, 0))
    draw_lane(lanes[1] * 1280, img, (255, 255, 255))
    draw_lane(lanes[2] * 1280, img, (0, 0, 0))
    draw_lane(lanes[3] * 1280, img, (0, 255, 0))
    cv2.imshow("test", img)
    cv2.waitKey(0)

show_image()
#make_video("test_video_1.mp4", "./TUSimple/test_set/clips/0530/1492626158152981904_0/", "output.mp4")