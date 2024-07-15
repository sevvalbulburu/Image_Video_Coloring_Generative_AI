import gan128_dependent
import gan128_independent
import os
import cv2
from create_dataset_dependent import sceneChangeDetector
import tensorflow_io as tfio
import numpy as np
from skimage.color import lab2rgb
from time import time
import warnings
warnings.filterwarnings("error")

IMAGE_SIZE = 128
detector = sceneChangeDetector()


def get_image(p):
    frame = cv2.imread(p)[:,:IMAGE_SIZE]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = np.expand_dims(gray_frame, axis=-1)
    gray_frame = np.repeat(gray_frame, 3, axis=-1)
    lab = tfio.experimental.color.rgb_to_lab(frame/255)
    l = lab[:, :, :1]
    return gray_frame, l, frame


def test_image(model, test_input, mode):
    prediction = model(np.array([test_input]), training=True)
    ab = prediction[0]*128
    # check if all values are in the range of [-128, 127]
    if np.any(ab < -128) or np.any(ab > 127):
        # make the values in the range of [-128, 127]
        ab = np.clip(ab, -128, 127)
    # # create an array of a**2 + b**2
    # a_b_square_sum = ab[0]**2 + ab[1]**2
    # maxx = np.max(a_b_square_sum)
    # print(maxx)
    predicted_lab = np.dstack((test_input[:, :, 0], ab))
    try:
        predicted_rgb = lab2rgb(predicted_lab)
    except UserWarning:
        print(mode + " model's output is out of range")
        return None, False
    predicted_rgb = lab2rgb(predicted_lab)
   # illuminant : {"A", "B", "C", "D50", "D55", "D65", "D75", "E"}, optional
    predicted = (predicted_rgb * 255).astype(np.uint8)
    predicted = cv2.cvtColor(predicted, cv2.COLOR_RGB2BGR)
    return predicted, True


test_folder = "/home/alperenlcr/bitirme/village_island_dataset_test_dependent/rgb/"
frames = sorted(os.listdir(test_folder))

# first_frame = [1000, 2000, 3000, 4000, 5000, 6000, 7000]
# video_length_frame = 200

save_folder = "/home/alperenlcr/bitirme/village_island_results/"

gray_prev, gray_current, l_gray_prev, l_gray_current = None, None, None, None
#first = True
video_writer = cv2.VideoWriter("aa_result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 3, (128, 128))
start_index = 6000

dep_times, ind_times, scene_times = [], [], []
for frame_idx in range(start_index, len(frames)):#i, i+video_length_frame
    gray_current, l_current, original = get_image(test_folder+frames[frame_idx])
    if frame_idx == start_index:
        st = time()
        prediction_current_bgr, status = test_image(gan128_independent.generator_independent, l_current, mode='independent')
        ind_times.append(time()-st)
        #first = False
    # check the similarity beetwen gray frames
    elif not detector.is_scene_change(gray_prev, gray_current):  # if there is no scene change
        # l_current+prev_bgr's ab channels
        prev_lab = tfio.experimental.color.rgb_to_lab(cv2.cvtColor(prediction_prev_bgr, cv2.COLOR_BGR2RGB)/255)
        prev_ab = prev_lab[:, :, 1:]/128
        inp = np.dstack((l_current[:, :, 0], prev_ab[:, :, 0], prev_ab[:, :, 1]))
        st = time()
        prediction_current_bgr, status = test_image(gan128_dependent.generator_dependent, inp, mode='dependent')
        dep_times.append(time()-st)
        if not status:
            st = time()
            prediction_current_bgr, status = test_image(gan128_independent.generator_independent, l_current, mode='independent')
            ind_times.append(time()-st)
    else:
        st = time()
        prediction_current_bgr, status = test_image(gan128_independent.generator_independent, l_current, mode='independent')
        ind_times.append(time()-st)
    if prediction_current_bgr is None:
        print("Error in frame: ", frame_idx)
        continue
    # gray in 3 channels
    gray = np.array(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
    gray = np.expand_dims(gray, axis=-1)
    gray = np.repeat(gray, 3, axis=-1)
    img = np.concatenate((gray, original, prediction_current_bgr), axis=1)
    # resize the image to see it clearly 4 times
    img = cv2.resize(img, (img.shape[1]*4, img.shape[0]*4))
    cv2.imshow("original-generated", img)
    cv2.waitKey(0)
    video_writer.write(prediction_current_bgr)
    # cv2.imwrite(save_folder+f"frame_{str(frame_idx).zfill(7)}.png", img)
    gray_prev, l_prev, prediction_prev_bgr = gray_current, l_current, prediction_current_bgr
    if frame_idx == 6300:
        break
print("Done")
print("Dependent model time in seconds: ", sum(dep_times)/len(dep_times))
print("Independent model time: ", sum(ind_times)/len(ind_times))
#print("Scene change detection time: ", sum(scene_times)/len(scene_times))
print(sum(dep_times)+sum(ind_times)+sum(scene_times))
video_writer.release()
"""
# 200 ms 200. frame 4. batch 8.image
print()
print()
frame_idx = 200
inp = list(test_dataset.take((frame_idx//BATCH_SIZE)+1).as_numpy_iterator())[-1][0][frame_idx-(frame_idx//BATCH_SIZE)*BATCH_SIZE]
tar = list(test_dataset.take((frame_idx//BATCH_SIZE)+1).as_numpy_iterator())[-1][1][frame_idx-(frame_idx//BATCH_SIZE)*BATCH_SIZE]
img = generate_image(generator_dependent, inp, tar)
cv2.imwrite(WORKDIR+"sil2_200.png", img)
print()
#generate_image(generator_dependent, input_image, target)

"""