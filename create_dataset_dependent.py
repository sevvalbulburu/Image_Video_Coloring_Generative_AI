import cv2
import os
import numpy as np


class sceneChangeDetector:
    """
    This class is used to detect scene changes between frames.
    params:
        None
    EXAMPLE USAGE:
    detector = scene_change_detector()
    detector.is_scene_change(img1, img2)
    """
    def is_scene_change(self, img1, img2):
        """
        This function is used to detect scene changes between two images.
        params:
            img1: first image
            img2: second image
        return:
            True if there is a scene change, False otherwise
        """
        self.img1 = img1
        self.img2 = img2
        hist = self.hist_diff()
        ssim_val = self.ssim()
        if hist < 0.997 and ssim_val < 0.5:
            return True
        return False


    def hist_diff(self):
        """
        This function is used to calculate histogram difference between two images.
        params:
            None
        return:
            histogram difference
        """
        def calculate_histogram(image):
            # Convert image to HSV color space
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Compute the histogram
            hist = cv2.calcHist([hsv_image], [0, 1], None, [50, 60], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist

        # Calculate histograms
        hist1 = calculate_histogram(self.img1)
        hist2 = calculate_histogram(self.img2)

        # Compare histograms
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return similarity


    def ssim(self):
        """
        This function is used to calculate SSIM between two images.
        params:
            None
        return:
            SSIM value
        """
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = self.img1.astype(np.float64)
        img2 = self.img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()


class Video2Dataset_Dependent:
    """
    This class is used to convert videos to dataset.
    Video continuous frames checked for scene changes.
    If there is not a scene change, frames are merged and saved.
    params:
        skip_ms: skip milliseconds between frames        
        square_size: size of frames
        video_folder: folder containing videos
        dataset_folder: folder to save dataset
    EXAMPLE USAGE:
    Video2Dataset_Dependent(skip_ms=200, square_size=128, \
        video_folder='/home/alperenlcr/bitirme/RGB_videos_test',\
            dataset_folder='/home/alperenlcr/bitirme/dataset_test_dependent')
    """
    def __init__(self, skip_ms, square_size, video_folder, dataset_folder):
        self.detector = sceneChangeDetector()
        self.skip_ms = skip_ms
        self.square_size = square_size
        self.video_folder = video_folder
        self.dataset_folder = dataset_folder
        self.video_paths = [v for v in os.listdir(video_folder) if v.endswith('.mp4')]
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)
        self.rgb_folder = os.path.join(self.dataset_folder, 'rgb')
        if not os.path.exists(self.rgb_folder):
            os.makedirs(self.rgb_folder)
        # Convert videos to dataset
        self.video2dataset()


    def video2dataset(self):
        """
        This function is used to convert videos to dataset.
        Video continuous frames checked for scene changes.
        If there is not a scene change, frames are merged and saved.
        params:
            None
        return:
            None
        """
        count = 0
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(os.path.join(self.video_folder, video_path))
            frame_skip = int(cap.get(cv2.CAP_PROP_FPS) * self.skip_ms / 1000)
            frame_prev = None
            frame_current = None
            h, w = cap.read()[1].shape[:2]
            padding = (w - h) // 2
            cap.set(cv2.CAP_PROP_POS_MSEC, 10000)    # skip first 10 seconds, its usually black
            while True:
                ret, frame = cap.read()
                # # check if the frame in the last 10 seconds
                # if cap.get(cv2.CAP_PROP_POS_MSEC) > (cap.get(cv2.CAP_PROP_FRAME_COUNT) - 10000):
                #     break
                if not ret:
                    break
                # add padding to make the frame square
                frame = cv2.copyMakeBorder(frame, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                frame = cv2.resize(frame, (self.square_size, self.square_size))
                if frame_prev is None:
                    frame_prev = frame
                    continue
                frame_current = frame
                if not self.detector.is_scene_change(frame_prev, frame_current):
                    merged = np.concatenate((frame_prev, frame_current), axis=1)
                    cv2.imwrite(os.path.join(self.rgb_folder, f'{str(count).zfill(7)}.jpg'), merged)
                frame_prev = frame_current
                for i in range(frame_skip):
                    cap.read()
                count += 1
                if count % 1000 == 0:
                    print(f'{count} frames are processed.')
            cap.release()


# video2dataset = Video2Dataset_Dependent(skip_ms=333, square_size=128,\
#                    video_folder='/home/alperenlcr/bitirme/villages_islands/train',\
#                        dataset_folder='/home/alperenlcr/bitirme/village_island_dataset_train_dependent')

# video2dataset = Video2Dataset_Dependent(skip_ms=200, square_size=128,\
#                    video_folder='/home/alperenlcr/bitirme/villages_islands/test',\
#                        dataset_folder='/home/alperenlcr/bitirme/village_island_dataset_test_dependent')
