import torch
from torch import nn
import math
import numpy as np
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import copy
import os

def get_gaussian_kernel(kernel_size=3, sigma=0.6, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, stride=1, padding=1, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def f(point):
    key_point = np.zeros((256, 256), np.uint8)
    cv2.fillPoly(key_point, point.reshape(1, 6, 2), color=255)


    TP = TN = FP = FN = 0
    TP = np.sum((key_point > 0) * (label > 0))
    FP = np.sum((key_point > 0) * (1 - (label > 0)))
    FN = np.sum((1 - (key_point > 0)) * (label > 0))
    TN = np.sum((1 - (key_point > 0)) * (1 - (label > 0)))


    iou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0
    return iou



def Get_edge(img):
    new = copy.deepcopy(img)
    row, col = img.shape
    for i in range(row - 1):
        for j in range(col - 1):
            if i - 1 > 0 & i + 1 < row - 1 & j - 1 > 0 & j + 1 < col - 1:
                if img[i][j - 1] and img[i][j + 1] and img[i - 1][j] and img[i + 1][j]:
                    new[i][j] = 0

    return new


def Get_Edge_position(img):
    row, col = img.shape
    row -= 1
    col -= 1
    new = copy.deepcopy(img)
    x = []
    y = []
    num = 0
    while (new == 0).all() == False:
        flag = 0
        for i in range(row):
            for j in range(col):
                if new[i][j]:
                    start_x = i
                    start_y = j
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                    flag = 1
                    break

        if flag == 0:
            break

        flage = 1
        while (flage):
            if start_y > 0 and new[start_x][start_y - 1]:
                start_x = start_x
                start_y = start_y - 1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            #            flage=1
            elif start_y > 0 and start_x < 255 and new[start_x + 1][start_y - 1]:
                start_x = start_x + 1
                start_y = start_y - 1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            elif start_x < 255 and new[start_x + 1][start_y]:
                start_x = start_x + 1
                start_y = start_y
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            elif start_x < 255 and start_y < 255 and new[start_x + 1][start_y + 1]:
                start_x = start_x + 1
                start_y = start_y + 1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            elif start_x > 0 and start_y > 0 and new[start_x - 1][start_y - 1]:
                start_x = start_x - 1
                start_y = start_y - 1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            elif start_x > 0 and new[start_x - 1][start_y]:
                start_x = start_x - 1
                start_y = start_y
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            elif start_x > 0 and start_y < 255 and new[start_x - 1][start_y + 1]:
                start_x = start_x - 1
                start_y = start_y + 1
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            elif start_y < 255 and new[start_x][start_y + 1]:
                start_y = start_y + 1
                start_x = start_x
                x.append(start_x)
                y.append(start_y)
                new[start_x][start_y] = 0
            #            flage=1
            else:
                flage = 0
    x = np.array(x)
    y = np.array(y)
    return x, y


if __name__ == '__main__':

    input_path = '/data1/xujiahao/Project/LB-UNet/data/isic2018/train/masks'
    output_path = '/data1/xujiahao/Project/LB-UNet/data/isic2018/train/points1'
    output_path2 = '/data1/xujiahao/Project/LB-UNet/data/isic2018/train/points2'
    
    files = sorted(os.listdir(input_path))
    for _id, file in enumerate(files):
        if os.path.exists(os.path.join(output_path2, file)):
            continue
        print(_id)
        label_path = os.path.join(input_path, file)
        # img = cv2.imread(label_path)
        img = (np.array(Image.open(label_path).convert('L')) > 0)
        label = img.copy()
        label = np.array(label, np.uint8)
        for r in range(len(img)):
            for l in range(len(img[0])):
                # print(img[r, l])
                if img[r, l] > 0.5:
                    label[r, l] = 255
                else:
                    label[r, l] = 0

        label = torch.tensor(label).reshape(1, 256, 256)
        label = label.float()
        # conv = get_gaussian_kernel(kernel_size=3, sigma=0.8, channels=1)
        # label = conv(label)
        label = label.detach().numpy()
        label = np.array(label.reshape(256, 256), np.uint8)

        img = Get_edge(img.astype('uint8'))
        col, row = img.shape
        Map = np.zeros_like(img)
        edgex, edgey = Get_Edge_position(img)

        label_ori = label.copy()
        contours, _ = cv2.findContours(label_ori, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0].squeeze(1)

        N = 300
        cross_rate = 0.1
        variation_rate = 0.05
        animals = []
        lst = []

        best_val = 0
        best_point = np.array([])
        for t in range(2000):
            index = np.random.randint(0, len(contours), size=6)
            index = sorted(index)
            point = np.array([contours[i] for i in index])
            iou = f(point)
            # print(t, iou)
            if iou > best_val:
                best_val = iou
                best_point = index

            lst.append((index, iou))

        print(best_val)

        lst = sorted(lst, key=lambda val: -val[1])
        for i in range(N):
            animals.append(lst[i][0])


        def get_fitness(animals):
            fitness = []
            for i in range(len(animals)):
                animals[i] = sorted(animals[i])
            for animal in animals:
                arr = np.array([contours[i] for i in animal])
                fitness.append(f(arr))
            return np.array(fitness)

        def select_animal(animals, fitness):
            idx = np.random.choice(np.arange(N), size=N, replace=True, p=(fitness / (np.sum(fitness) + 1e-8)))
            new_animals = []
            for id in idx:
                new_animals.append(animals[id])
            return new_animals

        def variation(child, variation_rate):
            child = sorted(child)
            new_child = []

            for i in range(len(child)):
                if np.random.rand() < variation_rate:
                    if i == 0:
                        if np.random.rand() < 0.5:
                            new_child.append(np.random.randint(0, child[i+1] + 1))
                        else:
                            new_child.append(np.random.randint(child[5], len(contours)))
                    elif i == 5:
                        if np.random.rand() < 0.5:
                            new_child.append(np.random.randint(0, child[0] + 1))
                        else:
                            new_child.append(np.random.randint(child[4], len(contours)))
                    else:
                        new_child.append(np.random.randint(child[i-1], child[i+1] + 1))
                else:
                    new_child.append(child[i])

            return new_child


        def crossover_and_variation(animals, cross_rate, is_randomise=True):
            new_animals = []
            for father in animals:
                child = father
                if np.random.rand() < cross_rate:
                    mother = animals[np.random.randint(N)]
                    if is_randomise:
                        cross_points = np.random.randint(low=0, high=2, size=6)
                        index = np.argwhere(cross_points == 1)
                        for id in index:
                            child[id[0]] = mother[id[0]]

                variation(child, variation_rate)
                new_animals.append(child)
            return new_animals


        def evaluation(animals):
            fitness = get_fitness(animals)
            id = np.argmax(fitness)
            return fitness[id], animals[id]

        for t in range(100):
            fitness = get_fitness(animals)
            selected_animals = select_animal(animals, fitness)
            animals = crossover_and_variation(selected_animals, cross_rate)
            iou, point =evaluation(animals)
            # print(t, iou, animals)
            if iou > best_val:
                best_val = iou
                best_point = point

        print(best_val)
        key_point = np.zeros((256, 256), np.uint8)
        key_point2 = np.zeros((256, 256), np.uint8)
        # cv2.fillPoly(key_point, np.array(contours[best_point]).reshape(1, 6, 2), color=255)
        # cv2.polylines(key_point, np.array(contours).reshape(1, -1, 2), isClosed=True, color=255, thickness=2)
        # for x, y in contours:
        #     for dx in range(-1, 2):
        #         for dy in range(-1, 2):
        #             if abs(dx) ** 2 + abs(dy) ** 2 <= 1 and x + dx >= 0 and x + dx < 256 and y + dy >= 0 and y + dy < 256:
        #                 if abs(dx) ** 2 + abs(dy) ** 2 == 0:
        #                     key_point[y + dy, x + dx] = 255
        #                 else:
        #                     key_point[y + dy, x + dx] = 125

        # for i in range(len(edgex)):
        #     key_point[edgex[i], edgey[i]] = label_ori[edgex[i], edgey[i]]
        # for x, y in np.array(contours[best_point]).reshape(6, 2):
        #     for dx in range(-5, 6):
        #         for dy in range(-5, 6):
        #             if abs(dx) ** 2 + abs(dy) ** 2 <= 25 and x + dx >= 0 and x + dx < 256 and y + dy >= 0 and y + dy < 256:
        #                 key_point[y + dy, x + dx] = label_ori[y + dy, x + dx]
        # cv2.imwrite(output_path + '/' + file, key_point)

        for i in range(len(edgex)):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if abs(dx) ** 2 + abs(dy) ** 2 <= 1 and edgex[i] + dx >= 0 and edgex[i] + dx < 256 and edgey[i] + dy >= 0 and edgey[i] + dy < 256:
                        key_point2[edgex[i] + dx, edgey[i] + dy] = label_ori[edgex[i] + dx, edgey[i] + dy]
        for x, y in np.array(contours[best_point]).reshape(6, 2):
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if abs(dx) ** 2 + abs(dy) ** 2 <= 25 and x + dx >= 0 and x + dx < 256 and y + dy >= 0 and y + dy < 256:
                        key_point2[y + dy, x + dx] = label_ori[y + dy, x + dx]
        cv2.imwrite(output_path2 + '/' + file, key_point2)
