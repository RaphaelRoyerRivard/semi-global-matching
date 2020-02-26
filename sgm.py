"""
python implementation of the semi-global matching algorithm from Stereo Processing by Semi-Global Matching
and Mutual Information (https://core.ac.uk/download/pdf/11134866.pdf) by Heiko Hirschmuller.

author: David-Alexandre Beaupre
date: 2019/07/12

edited by: Raphael Royer-Rivard
date: 2020/02/19
"""

import argparse
import sys
import time as t
import os

import cv2
from skimage.feature import hog, BRIEF
from sklearn.utils.extmath import cartesian
from scipy.spatial.distance import cdist
import numpy as np


class Direction:
    def __init__(self, direction=(0, 0), name='invalid'):
        """
        represent a cardinal direction in image coordinates (top left = (0, 0) and bottom right = (1, 1)).
        :param direction: (x, y) for cardinal direction.
        :param name: common name of said direction.
        """
        self.direction = direction
        self.name = name


# 8 defined directions for sgm
N = Direction(direction=(0, -1), name='north')
NE = Direction(direction=(1, -1), name='north-east')
E = Direction(direction=(1, 0), name='east')
SE = Direction(direction=(1, 1), name='south-east')
S = Direction(direction=(0, 1), name='south')
SW = Direction(direction=(-1, 1), name='south-west')
W = Direction(direction=(-1, 0), name='west')
NW = Direction(direction=(-1, -1), name='north-west')


class Paths:
    def __init__(self):
        """
        represent the relation between the directions.
        """
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]


class Parameters:
    def __init__(self, max_disparity=64, P1_ratio=0.5, P2_ratio=6, csize=(7, 7), bsize=(3, 3), descriptor='BRIEF', BRIEF_descriptor_size=128, HOG_orientations=9, folder="."):
        """
        represent all parameters used in the sgm algorithm.
        :param max_disparity: maximum distance between the same pixel in both images.
        :param P1_ratio: penalty ratio of average cost for disparity difference = 1
        :param P2_ratio: penalty ratio of average cost for disparity difference > 1
        :param csize: size of the kernel for the census transform (or cell size for the HOG descriptor).
        :param bsize: size of the kernel for blurring the images and median filtering.
        :param descriptor: BRIEF, HOG or census
        :param BRIEF_descriptor_size: descriptor size to use with BRIEF
        :param HOG_orientations: number of orientations to use with HOG
        :param folder: folder in which to save the images
        """
        self.max_disparity = max_disparity
        self.P1_ratio = P1_ratio
        self.P2_ratio = P2_ratio
        self.csize = csize
        self.bsize = bsize
        self.descriptor = descriptor
        self.BRIEF_descriptor_size = BRIEF_descriptor_size
        self.HOG_orientations = HOG_orientations
        self.folder = folder


def load_images(left_name, right_name, parameters):
    """
    read and blur stereo image pair.
    :param left_name: name of the left image.
    :param right_name: name of the right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: blurred left and right images.
    """
    left = cv2.imread(left_name, 0)
    left = cv2.GaussianBlur(left, parameters.bsize, 0, 0)
    right = cv2.imread(right_name, 0)
    right = cv2.GaussianBlur(right, parameters.bsize, 0, 0)
    return left, right


def get_indices(offset, dim, direction, height):
    """
    for the diagonal directions (SE, SW, NW, NE), return the array of indices for the current slice.
    :param offset: difference with the main diagonal of the cost volume.
    :param dim: number of elements along the path.
    :param direction: current aggregation direction.
    :param height: H of the cost volume.
    :return: arrays for the y (H dimension) and x (W dimension) indices.
    """
    y_indices = []
    x_indices = []

    for i in range(0, dim):
        if direction == SE.direction:
            if offset < 0:
                y_indices.append(-offset + i)
                x_indices.append(0 + i)
            else:
                y_indices.append(0 + i)
                x_indices.append(offset + i)

        if direction == SW.direction:
            if offset < 0:
                y_indices.append(height + offset - i)
                x_indices.append(0 + i)
            else:
                y_indices.append(height - i)
                x_indices.append(offset + i)

    return np.array(y_indices), np.array(x_indices)


def get_path_cost(slice, offset, P1, P2):
    """
    part of the aggregation step, finds the minimum costs in a D x M slice (where M = the number of pixels in the
    given direction)
    :param slice: M x D array from the cost volume.
    :param offset: ignore the pixels on the border.
    :param P1: penalty for disparity difference = 1
    :param P2: penalty for disparity difference > 1
    :return: M x D array of the minimum costs for a given slice in a given direction.
    """
    other_dim = slice.shape[0]
    disparity_dim = slice.shape[1]

    disparities = [d for d in range(disparity_dim)] * disparity_dim
    disparities = np.array(disparities).reshape(disparity_dim, disparity_dim)

    penalties = np.zeros(shape=(disparity_dim, disparity_dim), dtype=slice.dtype)
    penalties[np.abs(disparities - disparities.T) == 1] = P1
    penalties[np.abs(disparities - disparities.T) > 1] = P2

    minimum_cost_path = np.zeros(shape=(other_dim, disparity_dim), dtype=slice.dtype)
    minimum_cost_path[offset - 1, :] = slice[offset - 1, :]

    for i in range(offset, other_dim):
        previous_cost = minimum_cost_path[i - 1, :]
        current_cost = slice[i, :]
        costs = np.repeat(previous_cost, repeats=disparity_dim, axis=0).reshape(disparity_dim, disparity_dim)
        costs = np.amin(costs + penalties, axis=0)
        minimum_cost_path[i, :] = current_cost + costs - np.amin(previous_cost)
    return minimum_cost_path


def aggregate_costs(cost_volume, parameters, paths):
    """
    second step of the sgm algorithm, aggregates matching costs for N possible directions (8 in this case).
    :param cost_volume: array containing the matching costs.
    :param parameters: structure containing parameters of the algorithm.
    :param paths: structure containing all directions in which to aggregate costs.
    :return: H x W x D x N array of matching cost for all defined directions.
    """
    height = cost_volume.shape[0]
    width = cost_volume.shape[1]
    disparities = cost_volume.shape[2]

    start = -(height - 1)
    end = width - 1

    average_cost = cost_volume.mean()
    P1 = parameters.P1_ratio * average_cost
    P2 = parameters.P2_ratio * average_cost
    print(f"average cost: {average_cost}, p1: {P1}, p2: {P2}")

    aggregation_volume = np.zeros(shape=(height, width, disparities, paths.size), dtype=cost_volume.dtype)

    path_id = 0
    for path in paths.effective_paths:
        print('\tProcessing paths {} and {}...'.format(path[0].name, path[1].name), end='')
        sys.stdout.flush()
        dawn = t.time()

        main_aggregation = np.zeros(shape=(height, width, disparities), dtype=aggregation_volume.dtype)
        opposite_aggregation = np.copy(main_aggregation)

        main = path[0]
        if main.direction == S.direction:
            for x in range(0, width):
                south = cost_volume[0:height, x, :]
                north = np.flip(south, axis=0)
                main_aggregation[:, x, :] = get_path_cost(south, 1, P1, P2)
                opposite_aggregation[:, x, :] = np.flip(get_path_cost(north, 1, P1, P2), axis=0)

        if main.direction == E.direction:
            for y in range(0, height):
                east = cost_volume[y, 0:width, :]
                west = np.flip(east, axis=0)
                main_aggregation[y, :, :] = get_path_cost(east, 1, P1, P2)
                opposite_aggregation[y, :, :] = np.flip(get_path_cost(west, 1, P1, P2), axis=0)

        if main.direction == SE.direction:
            for offset in range(start, end):
                south_east = cost_volume.diagonal(offset=offset).T
                north_west = np.flip(south_east, axis=0)
                dim = south_east.shape[0]
                y_se_idx, x_se_idx = get_indices(offset, dim, SE.direction, None)
                y_nw_idx = np.flip(y_se_idx, axis=0)
                x_nw_idx = np.flip(x_se_idx, axis=0)
                main_aggregation[y_se_idx, x_se_idx, :] = get_path_cost(south_east, 1, P1, P2)
                opposite_aggregation[y_nw_idx, x_nw_idx, :] = get_path_cost(north_west, 1, P1, P2)

        if main.direction == SW.direction:
            for offset in range(start, end):
                south_west = np.flipud(cost_volume).diagonal(offset=offset).T
                north_east = np.flip(south_west, axis=0)
                dim = south_west.shape[0]
                y_sw_idx, x_sw_idx = get_indices(offset, dim, SW.direction, height - 1)
                y_ne_idx = np.flip(y_sw_idx, axis=0)
                x_ne_idx = np.flip(x_sw_idx, axis=0)
                main_aggregation[y_sw_idx, x_sw_idx, :] = get_path_cost(south_west, 1, P1, P2)
                opposite_aggregation[y_ne_idx, x_ne_idx, :] = get_path_cost(north_east, 1, P1, P2)

        aggregation_volume[:, :, :, path_id] = main_aggregation
        aggregation_volume[:, :, :, path_id + 1] = opposite_aggregation
        path_id = path_id + 2

        dusk = t.time()
        print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return aggregation_volume


def compute_costs(left, right, parameters, save_images):
    """
    first step of the sgm algorithm, matching cost based on the chosen descriptor
        A) census transform (BRIEF) and hamming distance
        B) HOG and SSD
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    assert left.shape[0] == right.shape[0] and left.shape[1] == right.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    descriptor = parameters.descriptor

    height = left.shape[0]
    width = left.shape[1]
    cheight = parameters.csize[0]
    cwidth = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity

    if descriptor == "BRIEF":
        brief_extractor = BRIEF(descriptor_size=parameters.BRIEF_descriptor_size, patch_size=cheight, mode='normal')
        img_dtype = np.uint8
        left_features = np.zeros(shape=(height, width, parameters.BRIEF_descriptor_size), dtype=np.bool)
        right_features = np.zeros(shape=(height, width, parameters.BRIEF_descriptor_size), dtype=np.bool)
    elif descriptor == "HOG":
        img_dtype = np.float
        left_features = np.zeros(shape=(height, width, parameters.HOG_orientations), dtype=np.float)
        right_features = np.zeros(shape=(height, width, parameters.HOG_orientations), dtype=np.float)
    else:
        img_dtype = np.uint8
        left_features = np.zeros(shape=(height, width), dtype=np.uint64)
        right_features = np.zeros(shape=(height, width), dtype=np.uint64)
    left_features_img = np.zeros(shape=(height, width), dtype=img_dtype)
    right_features_img = np.zeros(shape=(height, width), dtype=img_dtype)

    print('\tComputing left and right features...', end='')
    sys.stdout.flush()
    dawn = t.time()
    if descriptor == 'BRIEF':
        pixels = cartesian([np.arange(height), np.arange(width)])
        # LEFT
        brief_extractor.extract(left, pixels)
        descriptors = brief_extractor.descriptors
        left_features[2:-3, 2:-3] = np.reshape(descriptors, (height - cheight + 2, width - cwidth + 2, left_features.shape[-1]))
        left_features_img[:] = left_features.sum(axis=-1)
        # RIGHT
        brief_extractor.extract(right, pixels)
        descriptors = brief_extractor.descriptors
        right_features[2:-3, 2:-3] = np.reshape(descriptors, (height - cheight + 2, width - cwidth + 2, right_features.shape[-1]))
        right_features_img[:] = right_features.sum(axis=-1)
    # pixels on the border will have no features
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            # LEFT
            image = left[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            if descriptor == 'HOG':
                left_features[y, x] = hog(image, parameters.HOG_orientations, pixels_per_cell=(cheight, cwidth), cells_per_block=(1, 1))
                left_features_img[y, x] = left_features[y, x].sum()
            elif descriptor == 'census':
                center_pixel = left[y, x]
                reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
                comparison = image - reference
                left_census = np.int64(0)
                for j in range(comparison.shape[0]):
                    for i in range(comparison.shape[1]):
                        if (i, j) != (y_offset, x_offset):
                            left_census = left_census << 1
                            if comparison[j, i] < 0:
                                bit = 1
                            else:
                                bit = 0
                            left_census = left_census | bit
                left_features_img[y, x] = np.uint8(left_census)
                left_features[y, x] = left_census

            # RIGHT
            image = right[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            if descriptor == 'HOG':
                right_features[y, x] = hog(image, parameters.HOG_orientations, pixels_per_cell=(cheight, cwidth), cells_per_block=(1, 1))
                right_features_img[y, x] = right_features[y, x].sum()
            elif descriptor == 'census':
                center_pixel = right[y, x]
                reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
                comparison = image - reference
                right_census = np.int64(0)
                for j in range(comparison.shape[0]):
                    for i in range(comparison.shape[1]):
                        if (i, j) != (y_offset, x_offset):
                            right_census = right_census << 1
                            if comparison[j, i] < 0:
                                bit = 1
                            else:
                                bit = 0
                            right_census = right_census | bit
                right_features_img[y, x] = np.uint8(right_census)
                right_features[y, x] = right_census

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    if save_images:
        if descriptor != "census":
            # Normalizing the summed features for visualization
            left_features_img = 255 * (left_features_img - left_features_img.min()).astype(np.float) / (left_features_img.max() - left_features_img.min()).astype(np.float)
            right_features_img = 255 * (right_features_img - right_features_img.min()).astype(np.float) / (right_features_img.max() - right_features_img.min()).astype(np.float)
        cv2.imwrite(f'{parameters.folder}/left_features.png', left_features_img)
        cv2.imwrite(f'{parameters.folder}/right_features.png', right_features_img)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()

    if descriptor == "BRIEF":
        cost_volume_dtype = np.uint16
        lfeatures = np.zeros(shape=(height, width, parameters.BRIEF_descriptor_size), dtype=np.bool)
        rfeatures = np.zeros(shape=(height, width, parameters.BRIEF_descriptor_size), dtype=np.bool)
    elif descriptor == "HOG":
        cost_volume_dtype = np.float
        lfeatures = np.zeros(shape=(height, width, parameters.HOG_orientations), dtype=np.float)
        rfeatures = np.zeros(shape=(height, width, parameters.HOG_orientations), dtype=np.float)
    else:
        cost_volume_dtype = np.uint32
        lfeatures = np.zeros(shape=(height, width), dtype=np.int64)
        rfeatures = np.zeros(shape=(height, width), dtype=np.int64)
    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=cost_volume_dtype)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=cost_volume_dtype)
    for d in range(0, disparity):
        # LEFT
        rfeatures[:, (x_offset + d):(width - x_offset)] = right_features[:, x_offset:(width - d - x_offset)]
        if descriptor == 'BRIEF':
            left_cost_volume[:, :, d] = np.count_nonzero(left_features != rfeatures, axis=-1)
        elif descriptor == "HOG":
            diff = left_features - rfeatures  # (H, W, orientations)
            left_cost_volume[:, :, d] = np.sqrt(np.sum(diff**2, 2))  # Summed Squared Difference along the last axis
        else:
            left_xor = np.int64(np.bitwise_xor(np.int64(left_features), rfeatures))
            left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
            while not np.all(left_xor == 0):
                tmp = left_xor - 1
                mask = left_xor != 0
                left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
                left_distance[mask] = left_distance[mask] + 1
            left_cost_volume[:, :, d] = left_distance

        # RIGHT
        lfeatures[:, x_offset:(width - d - x_offset)] = left_features[:, (x_offset + d):(width - x_offset)]
        if descriptor == 'BRIEF':
            right_cost_volume[:, :, d] = np.count_nonzero(right_features != lfeatures, axis=-1)
        elif descriptor == 'HOG':
            diff = right_features - lfeatures  # (H, W, orientations)
            right_cost_volume[:, :, d] = np.sqrt(np.sum(diff**2, 2))  # Summed Squared Difference along the last axis
        else:
            right_xor = np.int64(np.bitwise_xor(np.int64(right_features), lfeatures))
            right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
            while not np.all(right_xor == 0):
                tmp = right_xor - 1
                mask = right_xor != 0
                right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
                right_distance[mask] = right_distance[mask] + 1
            right_cost_volume[:, :, d] = right_distance

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume


def select_disparity(aggregation_volume):
    """
    last step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    disparity_map = np.argmin(volume, axis=2)
    return disparity_map


def normalize(volume, parameters):
    """
    transforms values from the range (0, 64) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 255.0 * volume / parameters.max_disparity


def get_recall(disparity, gt_path, args):
    """
    computes the recall of the disparity map.
    :param disparity: disparity image.
    :param gt_path: path to ground-truth image.
    :param args: program arguments.
    :return: rate of correct predictions.
    """
    gt = np.float32(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
    gt = np.int16(gt / 255.0 * float(args.disp))  # between 0 and args.disp (64)
    disparity = np.int16(np.float32(disparity) / 255.0 * float(args.disp))  # between 0 and args.disp (64)
    diff = np.abs(disparity - gt)
    correct = np.count_nonzero(diff <= 3)
    cv2.imwrite(f'{args.output}/{gt_path.split("/")[-1].split(".")[0]}_diff.png', diff / float(args.disp) * 255)
    cv2.imwrite(f'{args.output}/{gt_path.split("/")[-1].split(".")[0]}_error.png', (diff > 3) * 255)
    return float(correct) / gt.size


def evaluate_disparity_map(disparity, gt_path, dataset, output_image_name, args):
    """
    Computes the recall and the BMPRE (Bad Matching Pixels Relative Error) of the disparity map.
    It takes into account the relative error of disparities above a threshold and scales them based on the ground truth
    depth so that an error x in the background counts for more than the same error x in the foreground.
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6491759
    :param disparity: disparity image.
    :param gt_path: path to ground-truth image.
    :param dataset: name of the dataset.
    :param output_image_name: path and name of the output images (diff and error).
    :param args: program arguments.
    :return: evaluation value of the disparity map (the lower the better).
    """
    gt = np.float32(cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE))
    print("gt", gt.min(), gt.mean(), gt.max())
    print("disparity", disparity.min(), disparity.mean(), disparity.max())
    if dataset.endswith("2003"):
        gt = np.int16(gt / 255.0 * float(args.disp))  # between 0 and args.disp (64)
    elif dataset.endswith("2005"):
        gt = np.int16(gt / 3.0)  # between 0 and maximum disparity of the image (all values are true pixel disparities)
    else:
        print("Error, unknown dataset")
        return 0, 0
    disparity = np.int16(np.float32(disparity) / 255.0 * float(args.disp))  # between 0 and args.disp (64)
    print("gt", gt.min(), gt.mean(), gt.max())
    print("disparity", disparity.min(), disparity.mean(), disparity.max())
    diff = np.abs(disparity - gt)
    cv2.imwrite(f'{output_image_name}_diff.png', diff / float(args.disp) * 255)
    cv2.imwrite(f'{output_image_name}_error.png', (diff > 3) * 255)
    recall = np.count_nonzero(np.abs(disparity - gt) <= 3)
    diff[np.where(diff <= 3)] = 0
    gt[np.where(gt == 0)] = np.iinfo(np.int16).max
    diff_relative_to_depth = diff / gt
    bmpre = diff_relative_to_depth.sum()
    return float(recall) / gt.size, float(bmpre) / gt.size


def sgm():
    """
    main function applying the semi-global matching algorithm.
    :return: void.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='.', help='input folder in which to search recursively for stereo pairs')
    parser.add_argument('--disp', default=64, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--images', default=True, type=bool, help='save intermediate representations')
    parser.add_argument('--eval', default=True, type=bool, help='evaluate disparity map with recall and BMPRE, both with 3 pixel threshold')
    parser.add_argument('--descriptor', default='BRIEF', help='descriptor method for cost calculation (BRIEF, HOG or census)')
    args = parser.parse_args()

    input_folder = args.input
    disparity = args.disp
    save_images = args.images
    evaluate = args.eval
    descriptor = args.descriptor

    dawn = t.time()

    for path, subfolders, files in os.walk(input_folder):
        if "sgm_results" in path.split("\\"):
            continue

        if len(files) != 4:
            continue

        gts = []
        views = []
        for file in files:
            (gts if file.startswith("disp") else views).append(file)
        if len(gts) != 2 or len(views) != 2:
            continue

        print(path)

        left_gt = path + "/" + gts[0]
        right_gt = path + "/" + gts[1]
        left_view = path + "/" + views[0]
        right_view = path + "/" + views[1]

        split_path = path.split('\\')
        dataset = split_path[-2]
        output_folder = f"sgm_results/{split_path[-1]}_{descriptor}"
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        parameters = Parameters(max_disparity=disparity, P1_ratio=0.5, P2_ratio=6, csize=(7, 7), bsize=(3, 3),
                                descriptor=descriptor, BRIEF_descriptor_size=128, HOG_orientations=9, folder=output_folder)
        paths = Paths()

        print('\nLoading images...')
        left, right = load_images(left_view, right_view, parameters)

        print('\nStarting cost computation...')
        left_cost_volume, right_cost_volume = compute_costs(left, right, parameters, save_images)
        if save_images:
            left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), parameters))
            cv2.imwrite(f'{output_folder}/disp_map_left_cost_volume.png', left_disparity_map)
            right_disparity_map = np.uint8(normalize(np.argmin(right_cost_volume, axis=2), parameters))
            cv2.imwrite(f'{output_folder}/disp_map_right_cost_volume.png', right_disparity_map)

        print('\nStarting left aggregation computation...')
        left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)
        print('\nStarting right aggregation computation...')
        right_aggregation_volume = aggregate_costs(right_cost_volume, parameters, paths)

        print('\nSelecting best disparities...')
        left_disparity_map = np.uint8(normalize(select_disparity(left_aggregation_volume), parameters))
        right_disparity_map = np.uint8(normalize(select_disparity(right_aggregation_volume), parameters))
        if save_images:
            cv2.imwrite(f'{output_folder}/left_disp_map_no_post_processing.png', left_disparity_map)
            cv2.imwrite(f'{output_folder}/right_disp_map_no_post_processing.png', right_disparity_map)

        print('\nApplying median filter...')
        left_disparity_map = cv2.medianBlur(left_disparity_map, parameters.bsize[0])
        right_disparity_map = cv2.medianBlur(right_disparity_map, parameters.bsize[0])
        cv2.imwrite(f'{output_folder}/left_disparity_map.png', left_disparity_map)
        cv2.imwrite(f'{output_folder}/right_disparity_map.png', right_disparity_map)

        if evaluate:
            f = open(f'{output_folder}/result.txt', 'w+')
            print('\nEvaluating left disparity map...')
            recall, bmpre = evaluate_disparity_map(left_disparity_map, left_gt, dataset, output_folder + "/left", args)
            print('\tRecall = {:.2f}%'.format(recall * 100.0))
            print('\tBMPRE = {:.5f}'.format(bmpre))
            f.write('{:.2f};{:.5f}\n'.format(recall * 100.0, bmpre))
            print('\nEvaluating right disparity map...')
            recall, bmpre = evaluate_disparity_map(right_disparity_map, right_gt, dataset, output_folder + "/right", args)
            print('\tRecall = {:.2f}%'.format(recall * 100.0))
            print('\tBMPRE = {:.5f}'.format(bmpre))
            f.write('{:.2f};{:.5f}\n'.format(recall * 100.0, bmpre))
            f.close()

    dusk = t.time()
    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))


if __name__ == '__main__':
    sgm()
