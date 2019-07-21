"""
Preprocessing stuff.
"""
import numpy as np
import cv2


def resize_img(img, scale_factor):
    new_size = (np.floor(np.array(img.shape[0:2]) * scale_factor)).astype(int)
    new_img = cv2.resize(img, (new_size[1], new_size[0]))
    # This is scale factor of [height, width] i.e. [y, x]
    actual_factor = [
        new_size[0] / float(img.shape[0]), new_size[1] / float(img.shape[1])
    ]
    return new_img, actual_factor


def scale_and_crop(image, scale, center, person_shape, ad, img_size, last_person):
    if abs(last_person[0]-scale) < 0.01 and abs(last_person[1]-center[0]) < 10 and abs(last_person[2]-center[1]) < 10 and abs(last_person[3]-person_shape[0]) < 10 and abs(last_person[4]-person_shape[1]) < 10:
        scale = last_person[0]
        center[0] = last_person[1]
        center[1] = last_person[2]
        person_shape[0] = last_person[3]
        person_shape[1] = last_person[4]
    else:
        last_person[0] = scale
        last_person[1] = center[0]
        last_person[2] = center[1]
        last_person[3] = person_shape[0]
        last_person[4] = person_shape[1]

    print(scale, center, person_shape)
    image_scaled, scale_factors = resize_img(image, scale)
    # Swap so it's [x, y]
    scale_factors = [scale_factors[1], scale_factors[0]]
    center_scaled = np.round(center * scale_factors).astype(np.int)
    shape_scaled = np.round(person_shape * scale_factors).astype(np.int)

    height = shape_scaled[1]
    width = shape_scaled[0]
    top = max(int(center_scaled[1] - (ad + 1) * height / 1.9), 0)
    bottom = min(int(center_scaled[1] + (ad + 1) * height / 1.9), image_scaled.shape[1])
    left = max(int(center_scaled[0] - (ad + 0.5) * width), 0)
    right = min(int(center_scaled[0] + (ad + 0.5) * width), image_scaled.shape[0])

    print("=================")
    print(top,bottom,left,right)
    #create a black use numpy,size is img_size*img_size
    mask = np.zeros([image_scaled.shape[0], image_scaled.shape[1]], np.uint8)

    #fill the image with white
    mask[top:bottom, left:right] = 255

    image_mask = cv2.add(image_scaled, np.zeros(np.shape(image_scaled), dtype=np.uint8), mask=mask)

    margin = int(img_size / 2)
    image_pad = np.pad(
        image_mask, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    center_pad = center_scaled + margin
    # figure out starting point
    start_pt = center_pad - margin
    end_pt = center_pad + margin
    # crop:
    crop = image_pad[start_pt[1]:end_pt[1], start_pt[0]:end_pt[0], :]
    proc_param = {
        'scale': scale,
        'start_pt': start_pt,
        'end_pt': end_pt,
        'img_size': img_size
    }

    return crop, proc_param, last_person
