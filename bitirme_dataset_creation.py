import cv2
import os
from random import random


class Video2Dataset:
    """
    This class is used to convert videos to dataset.
    Videos are splitted into frames with a given period.
    Frames are resized to given height and width.
    Frames are saved as jpg images.
    Dataset is splitted into train and test sets with a given ratio.
    params:
        train_test_split_ratio: ratio of train set size to dataset size
        period: period of frames to be saved
        square_size: size of frames
        video_folder: folder containing videos
        dataset_folder: folder to save dataset
    EXAMPLE USAGE:
    Video2Dataset(train_test_split_ratio=0.8, period=10, square_size=256, \
              video_folder='/home/alperenlcr/Documents/stock', dataset_folder='/home/alperenlcr/bitirme_dataset')
    """
    def __init__(self, train_test_split_ratio, period, square_size, video_folder, dataset_folder):
        self.train_test_split = train_test_split_ratio
        self.period = period
        self.square_size = square_size
        self.video_folder = video_folder
        self.dataset_folder = dataset_folder
        self.video_paths = [v for v in os.listdir(video_folder) if v.endswith('.mp4')]
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        self.train_folder = os.path.join(dataset_folder, 'train')
        self.test_folder = os.path.join(dataset_folder, 'test')
        if not os.path.exists(self.train_folder):
            os.makedirs(self.train_folder)
        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)
        self.train_folder_rgb = os.path.join(self.train_folder, 'rgb')
        #self.train_folder_gray = os.path.join(self.train_folder, 'gray')
        self.test_folder_rgb = os.path.join(self.test_folder, 'rgb')
        #self.test_folder_gray = os.path.join(self.test_folder, 'gray')
        if not os.path.exists(self.train_folder_rgb):
            os.makedirs(self.train_folder_rgb)
        #if not os.path.exists(self.train_folder_gray):
        #    os.makedirs(self.train_folder_gray)
        if not os.path.exists(self.test_folder_rgb):
            os.makedirs(self.test_folder_rgb)
        #if not os.path.exists(self.test_folder_gray):
        #    os.makedirs(self.test_folder_gray)
        # Convert videos to dataset
        self.video2dataset()


    def video2dataset(self):
        image_count = 0
        for video_path in self.video_paths:
            video = cv2.VideoCapture(os.path.join(self.video_folder, video_path))
            fps = video.get(cv2.CAP_PROP_FPS)
            save_period = int(fps * self.period)
            frame_number = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                if frame_number % save_period == 0:
                    image_count += 1
                    # add padding to make the frame square
                    h, w, _ = frame.shape
                    padding = (w - h) // 2
                    frame = cv2.copyMakeBorder(frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                    frame = cv2.resize(frame, (self.square_size, self.square_size))
                    if random() < self.train_test_split:
                        # fill 6 digits with zeros
                        cv2.imwrite(os.path.join(self.train_folder_rgb, video_path.split('.')[0] + '_frame_' + str(image_count).zfill(6) + '.jpg'), frame)
                        #cv2.imwrite(os.path.join(self.train_folder_gray, video_path.split('.')[0] + '_frame_' + str(image_count).zfill(6) + '.jpg'), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    else:
                        cv2.imwrite(os.path.join(self.test_folder_rgb, video_path.split('.')[0] + '_frame_' + str(image_count).zfill(6) + '.jpg'), frame)
                        #cv2.imwrite(os.path.join(self.test_folder_gray, video_path.split('.')[0] + '_frame_' + str(image_count).zfill(6) + '.jpg'), cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                    if image_count % 100 == 0:
                        print("Saving {}. image".format(image_count))
                frame_number += 1
            video.release()


# Video2Dataset(train_test_split_ratio=1, period=1.5, square_size=128, \
#               video_folder='/home/alperenlcr/bitirme/villages_islands/train', dataset_folder='/home/alperenlcr/bitirme/village_dataset_independent')

# Video2Dataset(train_test_split_ratio=0, period=0.33, square_size=128, \
#               video_folder='/home/alperenlcr/bitirme/villages_islands/test', dataset_folder='/home/alperenlcr/bitirme/village_dataset_independent')
