import sys

gsam_root = '/home/albert/github/Grounded-Segment-Anything'
sys.path.append(gsam_root)

import os

import json
import torch
from PIL import Image

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import (
    build_sam,
    SamPredictor
)
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)


def save_mask_data(output_dir, mask_list, box_list, label_list):
    value = 0  # 0 for background

    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0] == True] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'mask.png'), bbox_inches="tight", dpi=300, pad_inches=0.0)

    json_data = [{
        'value': value,
        'label': 'background'
    }]
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split('(')
        logit = logit[:-1]  # the last is ')'
        json_data.append({
            'value': value,
            'label': name,
            'logit': float(logit),
            'box': box.numpy().tolist(),
        })
    with open(os.path.join(output_dir, 'mask.json'), 'w') as f:
        json.dump(json_data, f)


# parameters
image_path = '../notebooks/cam1_rgb.jpg'
grouneded_checkpoint = os.path.join(gsam_root, 'groundingdino_swint_ogc.pth')
sam_checkpoint = os.path.join(gsam_root, 'sam_vit_h_4b8939.pth')
box_threshold = 0.3
text_threshold = 0.25
config_file = os.path.join(gsam_root, 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py')
grounded_checkpoint = os.path.join(gsam_root, 'groundingdino_swint_ogc.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TEXT_PROMPT = 'objects'


def load_models():
    """
    Load the dino and SAM models.
    :return:
    """
    model = load_model(config_file, grounded_checkpoint, device=device)
    predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    return model, predictor


def norm_cv2_image(cv2_image):
    """
    Given an unnormalized RGB cv2 image, convert it so that it can be taken by the model.
    :param cv2_image: an unnormalized RGB image, e.g., one read from ROS topic
    :return: normalized image
    """
    # cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    normed_image, _ = transform(Image.fromarray(cv2_image), None)  # 3, h, w
    return normed_image, cv2_image


def predict(model, predictor, cv2_image, text_prompt=TEXT_PROMPT):
    """
    Predicts detection and segmentaion from the PIL image.
    :param model: the DINO model
    :param predictor: the SAM model
    :param cv2_image: a cv2 RGB image, e.g., directly read from ROS color topic by
    bridge = CvBridge(); bridge.imgmsg_to_cv2(ros_msg, desired_encoding='passthrough')
    :return: the segmentation mask, bounding boxes, and the corresponding labels
    """
    normed_image, _ = norm_cv2_image(cv2_image)
    boxes_filt, pred_phrases = get_grounding_output(
        model, normed_image, text_prompt, box_threshold, text_threshold, device=device
    )
    predictor.set_image(cv2_image)  # set the unnormed image

    # process the boxes
    size = cv2_image.shape
    H, W = size[0], size[1]  # (h, w) for cv2, but (w, h) for pil
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]

    # apply boxes
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, [H, W]).to(device)

    # SAM predicts
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    return masks, boxes_filt, pred_phrases


def plot_masks_only(masks):
    """
    Plot the masks in a blank background.
    :param masks: the masks predicted by the SAM model
    :return: None. Plots a mask with matplotlib.pyplot.
    """
    for mask in masks:
        show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)


def get_mask_image(masks, random_color=True):
    """
    Returns the mask image corresponding to the masks from SAM.
    :param masks: an np array given by the SAM model
    :param random_color: whether to use different and random colors for different objects
    :return: A np array image
    """
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    mask_images = []
    for mask in masks:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        mask_images.append(mask_image)

    mask_image_rgb = 1 - sum(mask_images)[..., :3]

    return mask_image_rgb


def draw_overlay_masks_boxes(image, masks, boxes_filt, pred_phrases, seed=0):
    """
    Returns an image that overlays segmentation masks and boxes on the image
    :param seed: seed for random number generator
    :param image: a cv2 image
    :param masks: optional
    :param boxes_filt: optional
    :return: overlaid image
    """
    np.random.seed(seed)

    # draw output image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    if masks is not None:
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

    if boxes_filt is not None and pred_phrases is not None:
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)


def cmyk_to_rgb(cmyk_image):
    if (cmyk_image > 1).any():
        # Scale the CMYK image to the range of 0 to 1
        cmyk_image = cmyk_image.astype(np.float32) / 255.0

    # Extract the CMYK channels
    c, m, y, k = cmyk_image[..., 0], cmyk_image[..., 1], cmyk_image[..., 2], cmyk_image[..., 3]

    # Calculate the RGB channels
    r = (1 - c) * (1 - k)
    g = (1 - m) * (1 - k)
    b = (1 - y) * (1 - k)

    # Stack the RGB channels to form the RGB image
    rgb_image = np.stack([r, g, b], axis=-1)

    # Convert the RGB image back to the range of 0 to 255
    rgb_image = (rgb_image * 255).astype(np.uint8)

    return rgb_image
