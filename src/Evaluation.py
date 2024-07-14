"""
Original code used for evaluation during training of STN and LPRNET models

"""

from model.LPRNET import LPRNet, CHARS
from model.STN import STNet
from data.load_data import LPRDataLoader, collate_fn
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt


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

    return labels, pred_labels



def eval(lprnet, STN, dataloader, dataset, device):
    lprnet = lprnet.to(device)
    STN = STN.to(device)
    total = 0
    for imgs, labels, lengths, lines in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        transfer = STN(imgs)
        # fixing evaluation for two line number plates
        if lines == "2":
            transfer1 = transfer.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
            width, height = transfer1.shape[:2]
            curr1 = transfer1[: width // 2, : height - height // 10]
            curr2 = transfer1[width // 2 :, height // 10 :]
            # print(curr1.shape, curr2.shape  )
            stacked = np.hstack((curr1, curr2))
            resized = cv2.resize(stacked, (94, 24))
            transposed = np.transpose(resized, (2, 0, 1))
            new_transfer = torch.from_numpy(transposed).float().unsqueeze(0).to(device)
            logits = lprnet(new_transfer)
        else:
            logits = lprnet(
                transfer
            )  # torch.Size([batch_size, CHARS length, output length ])

        preds = logits.cpu().detach().numpy()
        _, pred_labels = decode(preds, CHARS)

        start = 0
        for i, length in enumerate(lengths):
            label = labels[start : start + length]
            start += length
            if np.array_equal(np.array(pred_labels[i]), label.cpu().numpy()):
                total += 1

    accuracy = total / len(dataset)

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPR Evaluation")
    parser.add_argument("--img_size", default=(94, 24), help="the image size")
    parser.add_argument(
        "--img_dirs", default="./data/ccpd_weather", help="the images path"
    )
    parser.add_argument("--dropout_rate", default=0.5, help="dropout rate.")
    parser.add_argument("--batch_size", default=128, help="batch size.")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    lprnet = LPRNet(class_num=len(CHARS), dropout_rate=args.dropout_rate)
    lprnet.to(device)
    lprnet.load_state_dict(
        torch.load(
            "src/weights/lprnet_Iter_043200_model.ckpt",
            map_location=lambda storage, loc: storage,
        )["net_state_dict"]
    )
    #    checkpoint = torch.load('saving_ckpt/lprnet_Iter_023400_model.ckpt')
    #    lprnet.load_state_dict(checkpoint['net_state_dict'])
    lprnet.eval()
    print("LPRNet loaded")

    #    torch.save(lprnet.state_dict(), 'weights/Final_LPRNet_model.pth')

    STN = STNet()
    STN.to(device)
    STN.load_state_dict(
        torch.load(
            "src/weights/stn_Iter_043200_model.ckpt",
            map_location=lambda storage, loc: storage,
        )["net_state_dict"]
    )
    STN.eval()
    print("STN loaded")
    dataset = LPRDataLoader([args.img_dirs], args.img_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    print("Dataset loaded with length : {}".format(len(dataset)))
    ACC = eval(lprnet, STN, dataloader, dataset, device)
    print("The accuracy is {:.2f} %".format(ACC * 100))
