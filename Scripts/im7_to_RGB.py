import ReadIM
import numpy as np
from PIL import Image, ImageOps


def im7_to_RGB(im7_path, Cam_number, invert=False): 

    vbuff1, vatts1 = ReadIM.extra.get_Buffer_andAttributeList(im7_path)
    v_array1, vbuff1 = ReadIM.extra.buffer_as_array(vbuff1)

    # Normalize the image data to 0-1, Convert the image data to uint8 format in the range 0-255
    v_array1_norm = (v_array1[Cam_number] - np.min(v_array1[Cam_number])) / (np.max(v_array1[Cam_number]) - np.min(v_array1[Cam_number]))
    v_array1_uint8 = (v_array1_norm * 255).astype(np.uint8)
    image_pil = Image.fromarray(v_array1_uint8).convert("RGB")
    if invert:
        image_pil = ImageOps.invert(image_pil)

    return image_pil