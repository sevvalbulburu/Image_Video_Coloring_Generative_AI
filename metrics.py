import math
import numpy as np
import cv2
import os


class CalculateMetrics:
    def __init__(self, path):
        self.path = path
        self.images_gt = []
        self.images_output = []
        self.load_images()


    def get_metrics(self):
        print("images loaded")
        ssim_min, ssim_max, ssim_mean = self.calculate_ssim()
        print("SSIM calculated")
        psnr_min, psnr_max, psnr_mean = self.calculate_psnr()
        print("PNSR calculated")
        return {
            "ssim": {
                "min": ssim_min,
                "max": ssim_max,
                "mean": ssim_mean
            },
            "psnr": {
                "min": psnr_min,
                "max": psnr_max,
                "mean": psnr_mean
            }
        }


    def load_images(self):
        ims = os.listdir(self.path)
        for im in ims:
            img = cv2.imread(os.path.join(self.path, im))
            # split the image horizontally into 3 parts
            im1, im2, im3 = np.hsplit(img, 3)
            self.images_gt.append(im2)
            self.images_output.append(im3)


    def ssim(self, img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
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


    def calculate_ssim(self):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''
        self.ssims_res = []
        for img1, img2 in zip(self.images_gt, self.images_output):
            if not img1.shape == img2.shape:
                raise ValueError('Input images must have the same dimensions.')
            if img1.ndim == 2:
                self.ssims_res.append(self.ssim(img1, img2))
            elif img1.ndim == 3:
                if img1.shape[2] == 3:
                    ssims = []
                    for i in range(3):
                        ssims.append(self.ssim(img1, img2))
                    self.ssims_res.append(np.array(ssims).mean())
                elif img1.shape[2] == 1:
                    self.ssims_res.append(self.ssim(np.squeeze(img1), np.squeeze(img2)))
            else:
                raise ValueError('Wrong input image dimensions.')
        return np.array(self.ssims_res).min(), np.array(self.ssims_res).max(), np.array(self.ssims_res).mean()


    def calculate_psnr(self):
        self.psnrs = []
        for img1, img2 in zip(self.images_gt, self.images_output):
            # img1 and img2 have range [0, 255]
            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            mse = np.mean((img1 - img2)**2)
            if mse == 0:
                raise ValueError('Mean Squared Error is zero. PSNR cannot be calculated.')
            self.psnrs.append(20 * math.log10(255.0 / math.sqrt(mse)))
        return np.array(self.psnrs).min(), np.array(self.psnrs).max(), np.array(self.psnrs).mean()


path = [
        # "<function generator1 at 0x7f762559dee0>-<function discriminator1 at 0x7f762559de40>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-128",
        # "<function generator1 at 0x7f93f3731480>-<function discriminator1 at 0x7f93f3731510>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-256",
        # "<function generator1 at 0x739f290f1ee0>-<function discriminator2 at 0x739f290f1d00>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-128",
        # "<function generator1 at 0x739f290f1ee0>-<function discriminator2 at 0x739f290f1d00>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-256",
        # "<function generator2 at 0x76a5626f2020>-<function discriminator1 at 0x76a5626f1e40>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-128",
        # "<function generator2 at 0x75df0e2453f0>-<function discriminator1 at 0x75df0e245510>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-256",
        # "<function generator2 at 0x75df0e2453f0>-<function discriminator2 at 0x75df0e245360>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-128",
        # "<function generator2 at 0x75df0e2453f0>-<function discriminator2 at 0x75df0e245360>-{'name': 'adam', 'learning_rate': 0.0002, 'beta_1': 0.5}-{'name': 'adam', 'learning_rate': 0.002, 'beta_1': 0.7}-256"
"/home/alperenlcr/Downloads/hababam_results",
"/home/alperenlcr/Downloads/village_island_results"
]
for p in path:
    metrics = CalculateMetrics(p).get_metrics()
    for key, value in metrics.items():
        #print(key, value)
        print(f"{value['min']:2f} & {value['max']:2f} & {value['mean']:2f}")


