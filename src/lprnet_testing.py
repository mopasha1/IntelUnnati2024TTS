import sys
import os
from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
import numpy as np
import argparse
import torch
import time
import cv2


sys.path.append(os.getcwd())  # adding current directory to path


def convert_image(input_image):
    # convert a Tensor to numpy image
    input_image = input_image.squeeze(0).cpu()
    input_image = input_image.detach().numpy().transpose((1, 2, 0))
    input_image = 127.5 + input_image / 0.0078125  # Denormalizeation
    input_image = input_image.astype("uint8")

    return input_image


def decode(preds, CHARS):
    """
    This function decodes the labels from the predictions tensor
    """
    pred_labels, labels = [], []
    n = preds.shape[0]
    
    for i in range(n):
        pred = preds[i, :, :]
        m = pred.shape[1]
        pred_label = []

        # using the maximum probability character as the predicted character

        for j in range(m):
            pred_label.append(np.argmax(pred[:, j], axis=0))
        non_repeated = [pred_label[0]]
        prev = pred_label[0]
        # Dropping repeated characters
        for c in pred_label[1:]:
            if (prev == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    prev = c
                continue
            non_repeated.append(c)
            prev = c
        pred_labels.append(non_repeated)

    for i, label in enumerate(pred_labels):
        lb = ""
        for i in label:
            lb += CHARS[i]
        labels.append(lb)

    return labels, np.array(pred_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR Demo")
    parser.add_argument("-image", help="image path", required=True, type=str)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=0)
    lprnet.to(device)
    lprnet.load_state_dict(
        torch.load(
            "./weights/lprnet_Iter_043200_model.ckpt",
            map_location=lambda storage, loc: storage,
        )["net_state_dict"]
    )
    lprnet.eval()

    STN = STNet()
    STN.to(device)
    STN.load_state_dict(
        torch.load(
            "weights/stn_Iter_043200_model.ckpt",
            map_location=lambda storage, loc: storage,
        )["net_state_dict"]
    )
    STN.eval()

    print("Successfully built the network!")

    since = time.time()
    image = cv2.imread(args.image)
    print(image.shape)
    im = cv2.resize(image, (94, 24))
    alpha = 1.5  # Contrast control
    beta = 5  # Brightness control

    # call convertScaleAbs function
    im = cv2.convertScaleAbs(im, alpha=alpha, beta=beta)
    # print(im.shape)
    im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * 0.0078125
    data = (
        torch.from_numpy(im).float().unsqueeze(0).to(device)
    )  # torch.Size([1, 3, 24, 94])
    transfer = STN(data)
    # original inference code, without fix for two line number plates. Fix in "model_inference.py"
    cv2.imshow("resized", transfer.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0))
    cv2.waitKey(0)
    preds = lprnet(transfer)
    preds = preds.cpu().detach().numpy()  # (1, 37, 18)

    labels, pred_labels = decode(preds, CHARS)
    print("model inference in {:2.3f} seconds".format(time.time() - since))
    print("Predicted: ", labels[0])
    print("Actual: ", args.image.split("\\")[-1].split(".")[0])
