import torch
from torch.utils.data import *
from imutils import paths
import numpy as np
import random
import cv2
import os


# List of characters in Indian License plates. - is used for blank character
CHARS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "J",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "I",
    "O",
    "-",
]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}


# Augmentation


def jitter(img, jitter=0.1):
    rows, cols, _ = img.shape
    j_width = float(cols) * random.uniform(1 - jitter, 1 + jitter)
    j_height = float(rows) * random.uniform(1 - jitter, 1 + jitter)
    img = cv2.resize(img, (int(j_width), int(j_height)))
    return img


def rotate(img, angle=5):
    scale = random.uniform(0.9, 1.1)
    angle = random.uniform(-angle, angle)
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    rotated = img.copy()
    rotated = cv2.warpAffine(img, M, (cols, rows), rotated, cv2.INTER_LINEAR)

    return rotated


def perspective(img):
    h, w, _ = img.shape
    per = random.uniform(0.05, 0.1)
    w_p = int(w * per)
    h_p = int(h * per)

    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pts2 = np.float32(
        [
            [random.randint(0, w_p), random.randint(0, h_p)],
            [random.randint(0, w_p), h - random.randint(0, h_p)],
            [w - random.randint(0, w_p), random.randint(0, h_p)],
            [w - random.randint(0, w_p), h - random.randint(0, h_p)],
        ]
    )

    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (w, h))
    return img


def crop_subimage(img, margin=2):
    ran_margin = random.randint(0, margin)
    rows, cols, _ = img.shape
    crop_h = rows - ran_margin
    crop_w = cols - ran_margin
    row_start = random.randint(0, ran_margin)
    cols_start = random.randint(0, ran_margin)
    sub_img = img[row_start : row_start + crop_h, cols_start : cols_start + crop_w]
    return sub_img


def hsv_space_variation(ori_img, scale):
    rows, cols, _ = ori_img.shape

    hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2HSV)
    hsv_img = np.array(hsv_img, dtype=np.float32)
    img = hsv_img[:, :, 2]

    # gaussian noise
    noise_std = random.randint(5, 20)
    noise = np.random.normal(0, noise_std, (rows, cols))

    # brightness scale
    img = img * scale
    img = np.clip(img, 0, 255)
    img = np.add(img, noise)

    # random hue variation
    hsv_img[:, :, 0] += random.randint(-5, 5)

    # random sat variation
    hsv_img[:, :, 1] += random.randint(-30, 30)

    hsv_img[:, :, 2] = img
    hsv_img = np.clip(hsv_img, 0, 255)
    hsv_img = np.array(hsv_img, dtype=np.uint8)
    rgb_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    return rgb_img


def data_augmentation(img):
    img = jitter(img)

    if random.choice([True, False]):
        img = rotate(img)

    if random.choice([True, False]):
        img = perspective(img)

    img = crop_subimage(img)

    bright_scale = random.uniform(0.6, 1.2)
    img_out = hsv_space_variation(img, scale=bright_scale)

    return img_out


# Credits to dataloader author Xuannan Xu
class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, PreprocFun=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        random.shuffle(self.img_paths)
        self.img_size = imgSize

        if PreprocFun is not None:
            self.PreprocFun = PreprocFun
        else:
            self.PreprocFun = self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        height, width, _ = Image.shape
        Image = data_augmentation(Image)
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)

        Image = self.PreprocFun(Image)

        basename = os.path.basename(filename)
        imgname, suffix = os.path.splitext(basename)
        line = imgname.split("_")[1].upper()
        imgname = imgname.split("-")[0].split("_")[0].upper()
        label = list()
        # print(imgname)
        for c in imgname:
            label.append(CHARS_DICT[c])

        # if len(label) == 8:
        #     if self.check(label) == False:
        #         print(imgname)
        #         assert 0, "Error label ^~^!!!"

        return Image, label, len(label), line

    def transform(self, img):
        img = img.astype("float32")
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))

        return img

    def check(self, label):
        if (
            label[2] != CHARS_DICT["D"]
            and label[2] != CHARS_DICT["F"]
            and label[-1] != CHARS_DICT["D"]
            and label[-1] != CHARS_DICT["F"]
        ):
            print("Error label, Please check!")
            return False
        else:
            return True


def collate_fn(batch):
    """
    Collate the entire batch provided into traninable format
    """
    imgs = []
    labels = []
    lengths = []
    lines = []
    for _, sample in enumerate(batch):
        img, label, length, line = sample
        img = cv2.resize(img.transpose(1, 2, 0), (94, 24)).transpose(2, 0, 1)
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
        lines.append(line)
    labels = np.asarray(labels).flatten().astype(np.float32)

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths, lines)


if __name__ == "__main__":
    dataset = LPRDataLoader(["validation"], (94, 24))
    dataloader = DataLoader(
        dataset, batch_size=128, shuffle=False, num_workers=2, collate_fn=collate_fn
    )
    print("Length {}".format(len(dataset)))
    for imgs, labels, lengths in dataloader:
        print("Shape of image batch", imgs.shape)
        print("Shape of label batch", labels.shape)
        print("Length of label", len(lengths))
        break
