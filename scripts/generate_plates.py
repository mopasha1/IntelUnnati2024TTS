from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os
import random
import numpy as np
import cv2
from imgaug import augmenters as iaa
import tqdm

def rotate(img, angle=5):

    scale = random.uniform(0.9, 1.1)
    angle = random.uniform(-angle, angle)

    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, scale)
    dst = img.copy()
    dst = cv2.warpAffine(img, M, (cols, rows), dst, cv2.INTER_LINEAR)

    return dst

def perspective(img):

    h, w, _ = img.shape
    per = random.uniform(0.05, 0.1)
    w_p = int(w * per)
    h_p = int(h * per)

    pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
    pts2 = np.float32([[random.randint(0, w_p), random.randint(0, h_p)],
                       [random.randint(0, w_p), h - random.randint(0, h_p)],
                       [w - random.randint(0, w_p), random.randint(0, h_p)],
                       [w - random.randint(0, w_p), h - random.randint(0, h_p)]])

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
    sub_img = img[row_start:row_start + crop_h, cols_start:cols_start + crop_w]
    return sub_img

def hsv_space_variation(ori_img, scale):

    rows, cols, _ = ori_img.shape

    hsv_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2HSV)
    hsv_img = np.array(hsv_img, dtype=np.float32)
    img = hsv_img[:, :, 2]

    # gau noise
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


    # if random.choice([True, False]):
    #     img = rotate(img)

    # if random.choice([True, False]):
    #     img = perspective(img)

    # img = crop_subimage(img)
    seq = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10)),  # Rotate
    iaa.AdditiveGaussianNoise(scale=(10, 30)),  # Add noise
    iaa.GaussianBlur(sigma=(0.0, 0.6)),  # Blur
    iaa.LinearContrast((0.4, 1.5)),  # Contrast
])

    # bright_scale = random.uniform(0.6, 1.2)
    # img_out = hsv_space_variation(img, scale=bright_scale)
    img_out = seq(image=img)

    return img_out
# Create a directory to save generated images
output_dir = r"generated_plates_new"
os.makedirs(output_dir, exist_ok=True)

# Load fonts
fonts = ["font1", "font2"] # Add paths to your license plate fonts

# License plate template
plate_template = "XX00XX0000"  # Adjust as per Indian license plate standards

# Function to generate random license plate text
def generate_plate_text():
    state_code = random.choice(
["AR","AS","BR","CG","DL","GA","GJ","HR","HP","JK","JH","KA","KL","LD","MP","MH","MN","ML","MZ","NL","OD","OR","PY","PB","RJ","SK","TN","TS","TR","UP","UK","UA","WB","AN","CH","DN","DD","LA","OT"]
)
    district_code = "".join(random.choices("0123456789", k=2))
    series_code = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
    number_code = "".join(random.choices("0123456789", k=4))
    filler = random.choice(["", " ", '.', '-'])
    return f"{state_code}{filler}{district_code}{filler}{series_code}{filler}{number_code}", f"{state_code}{district_code}{series_code}{number_code}"

# Function to create a synthetic license plate image
def create_license_plate():
    plate_text, plain_text = generate_plate_text()
    font = ImageFont.truetype(random.choice(fonts), 10)
    hm = {(255, 255, 255): (0, 0, 0), (255, 254, 0): (0, 0, 0), (0,0,0): (255,254,0), (58, 168, 54): (255,255,255)}
    x = random.choice(list(hm.keys())+[(255, 255, 255),(255, 255, 255)])
    image = Image.new("RGB", (94, 24), x)
    draw = ImageDraw.Draw(image)
    bbox = font.getbbox(plate_text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (94 - text_width) // 2
    text_y = (24 - text_height) // 2
    draw.text((text_x, text_y), plate_text, font=font, align='center', fill=hm[x])
    return image, plain_text

# Generate and save images
num_images = 25000
for i in tqdm.tqdm(range(num_images)):
    image , text = create_license_plate()
    new_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # new_image = data_augmentation(new_image)
    new_image = cv2.resize(new_image, (94, 24))
    cv2.imwrite(os.path.join(output_dir, f"{text}.jpg"), new_image)
    # image.save(os.path.join(output_dir, f"{text}.jpg"), "JPEG")

print(f"Generated {num_images} license plate images.")
