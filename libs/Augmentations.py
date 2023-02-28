import torch
import random
import math
import numpy as np
import scipy.ndimage
from skimage.transform import resize
import numpy.ma as ma


def norm95(image):
    # calculate the historgram of intensities of image and keep the top 95% then use mean of those to calculate
    # sort the array of intensities:
    image95 = image.flatten()
    size = len(image95)
    image95 = sorted(image95)[int(math.ceil((size * 5) / 100))-1:]
    image = (image - np.nanmean(image95) + 1e-10) / (np.nanstd(image95) + 1e-10)
    return image


class RandomZoom(object):
    # Zoom in augmentation
    # We zoom out the foreground parts when labels are available
    # We also zoom out the slices in the start and the end
    def __init__(self,
                 zoom_ratio_h=(0.5, 0.8),
                 zoom_ratio_w=(0.5, 0.8),
                 debug=0):

        self.zoom_ratio_h = zoom_ratio_h
        self.zoom_ratio_w = zoom_ratio_w
        self.debug = debug

    def sample_positions(self, image):
        ratio_h = round(random.uniform(self.zoom_ratio_h[0], self.zoom_ratio_h[1]), 2)
        ratio_w = round(random.uniform(self.zoom_ratio_w[0], self.zoom_ratio_w[1]), 2)
        # get image size upper bounds:
        h, w = np.shape(image)[-2], np.shape(image)[-1]
        # get cropping upper bounds:
        upper_h, upper_w = int(h*(1-ratio_h)), int(w*(1-ratio_w))
        # sampling positions:
        sample_h, sample_w = random.randint(0, upper_h), random.randint(0, upper_w)
        # sampling sizes:
        size_h, size_w = int(h * ratio_h), int(w * ratio_w)
        return sample_h, sample_w, size_h, size_w, ratio_h, ratio_w

    def sample_patch(self, image, label):

        h0, w0, new_h, new_w, ratio_h, ratio_w = self.sample_positions(label)
        cropped_image = image[h0:h0 + int(new_h), w0:w0 + int(new_w)]
        cropped_label = label[h0:h0 + int(new_h), w0:w0 + int(new_w)]

        # upsample them:
        zoomed_image = scipy.ndimage.zoom(input=cropped_image, zoom=(math.ceil(1 / ratio_h), math.ceil(1 / ratio_w)), order=1)
        zoomed_label = scipy.ndimage.zoom(input=cropped_label, zoom=(math.ceil(1 / ratio_h), math.ceil(1 / ratio_w)), order=0)

        return zoomed_image, zoomed_label

    def forward(self, image, label):
        image, label = np.squeeze(image), np.squeeze(label)
        image_zoomed, label_zoomed = self.sample_patch(image, label)

        # crop again to makes the zoomed image has the same size as the original image size:
        h, w = np.shape(label)[-2], np.shape(label)[-1]
        image_zoomed, label_zoomed = image_zoomed[0:h, 0:w], label_zoomed[0:h, 0:w]
        h2, w2 = np.shape(label_zoomed)[-2], np.shape(label_zoomed)[-1]

        assert h2 == h
        assert w2 == w

        return image_zoomed, label_zoomed


class RandomCrop(object):
    def __init__(self,
                 output_shape
                 ):
        self.output_shape = output_shape

    def crop_x(self, x):

        sample_position_d = random.randint(0, np.shape(x)[-3] - 1 - self.output_shape[0])
        sample_position_h = random.randint(0, np.shape(x)[-2] - 1 - self.output_shape[1])
        sample_position_w = random.randint(0, np.shape(x)[-1] - 1 - self.output_shape[2])

        if len(np.shape(x)) == 3:

            x = x[sample_position_d:sample_position_d+self.output_shape[0],
                  sample_position_h:sample_position_h+self.output_shape[1],
                  sample_position_w:sample_position_w+self.output_shape[2]]

        elif len(np.shape(x)) == 4:

            x = x[:,
                sample_position_d:sample_position_d + self.output_shape[0],
                sample_position_h:sample_position_h + self.output_shape[1],
                sample_position_w:sample_position_w + self.output_shape[2]]

        else:
            raise NotImplementedError

        return x

    def crop_xy(self, x, y):

        sample_position_d = random.randint(0, np.shape(x)[-3] - 1 - self.output_shape[0])
        sample_position_h = random.randint(0, np.shape(x)[-2] - 1 - self.output_shape[1])
        sample_position_w = random.randint(0, np.shape(x)[-1] - 1 - self.output_shape[2])

        if len(np.shape(x)) == 3:

            x = x[sample_position_d:sample_position_d+self.output_shape[0],
                  sample_position_h:sample_position_h+self.output_shape[1],
                  sample_position_w:sample_position_w+self.output_shape[2]]

            y = y[sample_position_d:sample_position_d+self.output_shape[0],
                  sample_position_h:sample_position_h+self.output_shape[1],
                  sample_position_w:sample_position_w+self.output_shape[2]]

        elif len(np.shape(x)) == 4:

            x = x[:, sample_position_d:sample_position_d + self.output_shape[0],
                sample_position_h:sample_position_h + self.output_shape[1],
                sample_position_w:sample_position_w + self.output_shape[2]]

            y = y[sample_position_d:sample_position_d + self.output_shape[0],
                sample_position_h:sample_position_h + self.output_shape[1],
                sample_position_w:sample_position_w + self.output_shape[2]]

        else:
            raise NotImplementedError

        return x, y


class RandomSlicingOrthogonal(object):
    def __init__(self,
                 output_size=(160, 160),
                 full_orthogonal=0,
                 # discarded_slices=5,
                 zoom=1,
                 # sampling_weighting_slope=0
                 ):
        '''
        cropping_d: 3 d dimension of cropped sub volume cropping on h x w
        cropping_h: 3 d dimension of cropped sub volume cropping on w x d
        cropping_w: 3 d dimension of cropped sub volume cropping on h x d
        '''
        # self.discarded_slices = discarded_slices
        self.zoom = zoom
        self.full_orthogonal = full_orthogonal

        self.new_size_h = output_size[0]
        self.new_size_w = output_size[1]

        # Over sampling the slices on the two ends because they contain small difficult vessels
        # We give the middle slice the lowest weight and the slices at the two very ends the highest weights

        # weight_middle_slice = 1
        # default_no_slices = 512
        # weight_end_slices = sampling_weighting_slope*0.5*default_no_slices + weight_middle_slice
        # sampling_weights_prob = [int(abs((i-0.5*default_no_slices))*sampling_weighting_slope+default_no_slices) / weight_end_slices for i in range(default_no_slices)]
        # self.sampling_weights_prob = [i / sum(sampling_weights_prob) for i in sampling_weights_prob] # normalise so everything adds up to 1

        if self.zoom == 1:
            self.zoom_aug = RandomZoom()

    def crop(self, *volumes):

        if self.full_orthogonal == 1:
            outputs = {"plane_d": [],
                       "plane_h": [],
                       "plane_w": []}

            d, h, w = np.shape(volumes[0])

            sample_position_d = random.randint(0, d - 1)
            sample_position_h = random.randint(0, h - 1)
            sample_position_w = random.randint(0, w - 1)

            for i, each_input in enumerate(volumes):
                if i == 0:
                    outputs["plane_d"].append(resize(np.squeeze(each_input[sample_position_d, :, :]), (self.new_size_h, self.new_size_w), order=1))
                    outputs["plane_h"].append(resize(np.squeeze(each_input[:, sample_position_h, :]), (self.new_size_h, self.new_size_w), order=1))
                    outputs["plane_w"].append(resize(np.squeeze(each_input[:, :, sample_position_w]), (self.new_size_h, self.new_size_w), order=1))
                elif i == 1:
                    outputs["plane_d"].append(resize(np.squeeze(each_input[sample_position_d, :, :]), (self.new_size_h, self.new_size_w), order=0))
                    outputs["plane_h"].append(resize(np.squeeze(each_input[:, sample_position_h, :]), (self.new_size_h, self.new_size_w), order=0))
                    outputs["plane_w"].append(resize(np.squeeze(each_input[:, :, sample_position_w]), (self.new_size_h, self.new_size_w), order=0))

            if self.zoom is True:
                if random.random() >= 0.5:
                    outputs["plane_d"][0], outputs["plane_d"][1] = self.zoom_aug.forward(outputs["plane_d"][0], outputs["plane_d"][1])
                    outputs["plane_h"][0], outputs["plane_h"][1] = self.zoom_aug.forward(outputs["plane_h"][0], outputs["plane_h"][1])
                    outputs["plane_w"][0], outputs["plane_w"][1] = self.zoom_aug.forward(outputs["plane_w"][0], outputs["plane_w"][1])

            return outputs

        elif self.full_orthogonal == 0:
            outputs = {"plane": []}

            d, h, w = np.shape(volumes[0])

            # sample_position_d = np.random.choice(np.arange(newd), 1, p=self.sampling_weights_prob)
            # sample_position_h = np.random.choice(np.arange(newh), 1, p=self.sampling_weights_prob)
            # sample_position_w = np.random.choice(np.arange(neww), 1, p=self.sampling_weights_prob)

            sample_position_d = random.randint(0, d - 1)
            sample_position_h = random.randint(0, h - 1)
            sample_position_w = random.randint(0, w - 1)

            roll_a_dice = random.random()

            if roll_a_dice < 0.34:
                for i, each_input in enumerate(volumes):
                    if i == 0:
                        outputs["plane"].append(resize(np.squeeze(each_input[sample_position_d, :, :]), (self.new_size_h, self.new_size_w), order=1))
                    else:
                        outputs["plane"].append(resize(np.squeeze(each_input[sample_position_d, :, :]), (self.new_size_h, self.new_size_w), order=0))

            elif roll_a_dice < 0.68:
                for i, each_input in enumerate(volumes):
                    if i == 0:
                        outputs["plane"].append(resize(np.squeeze(each_input[:, sample_position_h, :]), (self.new_size_h, self.new_size_w), order=1))
                    else:
                        outputs["plane"].append(resize(np.squeeze(each_input[:, sample_position_h, :]), (self.new_size_h, self.new_size_w), order=0))

            else:
                for i, each_input in enumerate(volumes):
                    if i == 0:
                        outputs["plane"].append(resize(np.squeeze(each_input[:, :, sample_position_w]), (self.new_size_h, self.new_size_w), order=1))
                    else:
                        outputs["plane"].append(resize(np.squeeze(each_input[:, :, sample_position_w]), (self.new_size_h, self.new_size_w), order=0))

            if self.zoom == 1:
                if random.random() >= 0.5:
                    outputs["plane"][0], outputs["plane"][1] = self.zoom_aug.forward(outputs["plane"][0], outputs["plane"][1])

            return outputs


class RandomContrast(object):
    def __init__(self, bin_range=(10, 100)):
        # self.bin_low = bin_range[0]
        # self.bin_high = bin_range[1]
        self.bin_range = bin_range

    def randomintensity(self, input):
        augmentation_flag = np.random.rand()
        if augmentation_flag >= 0.5:
            bin = random.randint(self.bin_range[0], self.bin_range[1])

            if len(np.shape(input)) == 3:
                c, h, w = np.shape(input)
                image_histogram, bins = np.histogram(input.flatten(), bin, density=True)
                cdf = image_histogram.cumsum()  # cumulative distribution function
                cdf = 255 * cdf / cdf[-1]  # normalize
                output = np.interp(input.flatten(), bins[:-1], cdf)
                output = np.reshape(output, (c, h, w))
            elif len(np.shape(input)) == 4:
                c, d, h, w = np.shape(input)
                image_histogram, bins = np.histogram(input.flatten(), bin, density=True)
                cdf = image_histogram.cumsum()  # cumulative distribution function
                cdf = 255 * cdf / cdf[-1]  # normalize
                output = np.interp(input.flatten(), bins[:-1], cdf)
                output = np.reshape(output, (c, d, h, w))
            else:
                raise NotImplementedError

        else:
            output = input

        return output


class RandomGaussian(object):
    def __init__(self, mean=0, std=0.1):
        self.m = mean
        self.sigma = std

    def gaussiannoise(self, input):
        noise = np.random.normal(self.m, self.sigma, input.shape)
        mask_overflow_upper = input + noise >= 1.0
        mask_overflow_lower = input + noise < 0.0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0.0
        input += noise
        return input


def randomcutout(x, y):
    '''
    Args:
        x: segmentation
        y: gt
    Returns:
    '''
    b, c, h, w = x.size()
    h_mask, w_mask = random.randint(int(h // 5), int(h // 2)), random.randint(int(w // 5), int(w // 2))

    h_starting = np.random.randint(0, h - h_mask)
    w_starting = np.random.randint(0, w - h_mask)
    h_ending = h_starting + h_mask
    w_ending = w_starting + w_mask

    mask = torch.ones_like(x).cuda()
    mask[:, :, h_starting:h_ending, w_starting:w_ending] = 0

    return x*mask, y*mask


# class RandomCutMix(object):
#     # In house implementation of cutmix for segmentation. This is inspired by the original cutmix but very different from the original one!!
#     # We mix a part of image 1 with another part of image 2 and do the same for the paired labels.
#     # This is applied before feeding into the network!!!
#     def __int__(self, tensor, mask_height=50, mask_width=50):
#
#         self.segmentation_tensor = tensor
#         self.w_mask = mask_width
#         self.h_mask = mask_height
#
#     def cutmix_seg(self, x, y):
#         '''
#         Args:
#             x: segmentation
#             y: gt
#         Returns:
#         '''
#         b, c, h, w = self.segmentation_tensor.size()
#
#         assert self.w_mask <= w
#         assert self.h_mask <= h
#
#         h_starting = np.random.randint(0, h - self.h_mask)
#         w_starting = np.random.randint(0, w - self.h_mask)
#         h_ending = h_starting + self.h_mask
#         w_ending = w_starting + self.w_mask
#
#         index = np.random.permutation(b)
#         x_2 = x[index, :]
#         y_2 = y[index, :]
#
#         x[:, :, h_starting:h_ending, w_starting:w_ending] = x_2[:, :, h_starting:h_ending, w_starting:w_ending]
#         y[:, :, h_starting:h_ending, w_starting:w_ending] = y_2[:, :, h_starting:h_ending, w_starting:w_ending]
#
#         return x, y