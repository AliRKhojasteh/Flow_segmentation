import os
import cv2
import torch
import numpy as np

from groundingdino.util import box_ops
from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import predict as dino_predict

from huggingface_hub import hf_hub_download

from segment_anything import sam_model_registry
from segment_anything import SamPredictor
import matplotlib.pyplot as plt

#######################################
########## Superimpose masks ##########
#######################################

def Superimpose_image_with_masks(image, masks):
    num_masks = len(masks)

    fig, axes = plt.subplots(1, num_masks + 1, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    image_np = np.array(image)

    for i, mask_np in enumerate(masks):
        # Convert the mask to an 8-bit array if it's a boolean array
        if mask_np.dtype == np.bool_:
            mask_np = mask_np.astype(np.uint8)

        color = np.array([0.9, 0.2, 0.1])  

        mask_color_bgr = (color[:3] * 255).astype(np.uint8)  # Convert to BGR and uint8
        mask_color_bgr_3d = np.ones_like(image_np) * mask_color_bgr
        superimposed_image = image_np.copy()
        superimposed_image[mask_np == 1] = mask_color_bgr

        alpha = 0.25
        superimposed_image = cv2.addWeighted(image_np, alpha, superimposed_image, 1 - alpha, 0)

        axes[i+1].imshow(superimposed_image, cmap='gray')
        axes[i+1].set_title(f"Mask {i+1}")
        axes[i+1].axis('off')

    plt.tight_layout()
    plt.show()

#######################################
########## Groundingdino model ########
#######################################

def Groundingdino_model(repo_id="ShilongLiu/GroundingDINO", filename="groundingdino_swinb_cogcoor.pth", config_filename="GroundingDINO_SwinB.cfg.py", device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)

    print("Check the latest .pth from: https://github.com/IDEA-Research/GroundingDINO/releases")

    model.eval()
    return model

#######################################
########## Segmentation model #########
#######################################

def Segmentation_model(sam_type="vit_b", ckpt_path=None, device='cpu'):
    Download_model = {
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_" + sam_type + "_4b8939.pth",
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_" + sam_type + "_0b3195.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_" + sam_type + "_01ec64.pth"
    }

    if ckpt_path is None:
        checkpoint = Download_model[sam_type]
        sam = sam_model_registry[sam_type]()
        state_dict = torch.hub.load_state_dict_from_url(checkpoint)
        sam.load_state_dict(state_dict, strict=True)
    else:
        sam = sam_model_registry[sam_type](ckpt_path)

    sam.to(device=device)
    sam = SamPredictor(sam)
    return sam

#######################################
########## Prediction functio #########
#######################################

def Predictor(image_pil, text_prompt, input_point, input_label, groundingdino_model, sam_model, aux_box, box_threshold=0.3, text_threshold=0.25):

    # Image transformation
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),

        # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]): 
        #This normalizes the image tensor with the provided mean and standard deviation for each color channel (RGB). 
        # These values are standard normalization parameters for images trained with ImageNet data. 
        # Normalization helps to standardize the input to a model and can lead to better performance.
    ])

    image_trans, _ = transform(image_pil, None)

    # GroundingDINO prediction
    boxes, logits, phrases = dino_predict(model=groundingdino_model,
                                          image=image_trans,
                                          caption=text_prompt,
                                          box_threshold=box_threshold,
                                          text_threshold=text_threshold,
                                          device=sam_model.device)
    W, H = image_pil.size
    boxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

    # SAM prediction
    image_array = np.asarray(image_pil)
    sam_model.set_image(image_array)

    if boxes.nelement() == 0:
        # If no boxes are detected, use the auxiliary box
        print("No boxes detected. Using the auxiliary box.")
        boxes = aux_box

    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes, image_array.shape[:2])

    if input_point is not None and input_label is not None:
        point_coords_tensor = torch.from_numpy(input_point).unsqueeze(0).to(sam_model.device)
        input_label_tensor = torch.from_numpy(input_label).unsqueeze(0).to(sam_model.device)
    else:
        point_coords_tensor = None
        input_label_tensor = None
        
    masks, _, _ = sam_model.predict_torch(
        point_coords=point_coords_tensor,
        point_labels=input_label_tensor,
        boxes=transformed_boxes.to(sam_model.device),
        multimask_output=False,
    )

    masks = masks.cpu()
    if len(boxes) > 0:
        masks = masks.squeeze(1)

    return masks, boxes, phrases, logits

#######################################
#######################################
#######################################