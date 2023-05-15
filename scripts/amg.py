# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2  # type: ignore

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import argparse
import json
import os
from typing import Any, Dict, List
import numpy as np
from tifffile import imread, imwrite
from skimage.util import img_as_ubyte, img_as_uint
import torch
from time import time 

class ChannelException(Exception):
    """Raise for trying to run the model on a multi-channel tiff without providing the channel that should be used."""

def convert_annotations_to_labels(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    labels = np.zeros_like(sorted_anns[0]['segmentation'], dtype='uint8')
    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        labels[mask] = i
    return labels

def load_image(image_path, channel = None):

    image = imread(image_path)
    if image.ndim > 2 and channel is None:
        raise ChannelException("When providing a multi-channel tiff, you need to specify which channel you want to run the model on using the argument --channel")
    elif image.ndim > 2:
        image = image[channel, ...] #type:ignore
    image_norm = img_as_uint(image/np.max(image))
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(8,8))
    image = img_as_ubyte(clahe.apply(image_norm))

    image = np.expand_dims(image, axis=2)
    image = np.repeat(image, 3, axis=2)
    return image

def get_args():
    parser = argparse.ArgumentParser(
    description=(
        "Runs automatic mask generation on an input image or directory of images, "
        "and outputs masks as label images."
    )
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to either a single input image or folder of images.",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help=(
            "Path to the directory where masks will be output."
        ),
    )

    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--channel",
        type=int,
        help="Choose the channel to run the segmentation on. Required if working with multi-channel tiff files.",
    )


    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )

    amg_settings = parser.add_argument_group("AMG Settings")

    amg_settings.add_argument(
        "--points-per-side",
        type=int,
        default=None,
        help="Generate masks by sampling a grid over the image with this many points to a side.",
    )

    amg_settings.add_argument(
        "--points-per-batch",
        type=int,
        default=None,
        help="How many input points to process simultaneously in one batch.",
    )

    amg_settings.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=None,
        help="Exclude masks with a predicted score from the model that is lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-thresh",
        type=float,
        default=None,
        help="Exclude masks with a stability score lower than this threshold.",
    )

    amg_settings.add_argument(
        "--stability-score-offset",
        type=float,
        default=None,
        help="Larger values perturb the mask more when measuring stability score.",
    )

    amg_settings.add_argument(
        "--box-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding a duplicate mask.",
    )

    amg_settings.add_argument(
        "--crop-n-layers",
        type=int,
        default=None,
        help=(
            "If >0, mask generation is run on smaller crops of the image to generate more masks. "
            "The value sets how many different scales to crop at."
        ),
    )

    amg_settings.add_argument(
        "--crop-nms-thresh",
        type=float,
        default=None,
        help="The overlap threshold for excluding duplicate masks across different crops.",
    )

    amg_settings.add_argument(
        "--crop-overlap-ratio",
        type=int,
        default=None,
        help="Larger numbers mean image crops will overlap more.",
    )

    amg_settings.add_argument(
        "--crop-n-points-downscale-factor",
        type=int,
        default=None,
        help="The number of points-per-side in each layer of crop is reduced by this factor.",
    )

    amg_settings.add_argument(
        "--min-mask-region-area",
        type=int,
        default=None,
        help=(
            "Disconnected mask regions or holes with area smaller than this value "
            "in pixels are removed by postprocessing."
        ),
    )

    return parser.parse_args()


def get_amg_kwargs(args):
    amg_kwargs = {
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "stability_score_offset": args.stability_score_offset,
        "box_nms_thresh": args.box_nms_thresh,
        "crop_n_layers": args.crop_n_layers,
        "crop_nms_thresh": args.crop_nms_thresh,
        "crop_overlap_ratio": args.crop_overlap_ratio,
        "crop_n_points_downscale_factor": args.crop_n_points_downscale_factor,
        "min_mask_region_area": args.min_mask_region_area,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    _ = sam.to(device=device)
    amg_kwargs = get_amg_kwargs(args)
    generator = SamAutomaticMaskGenerator(sam, **amg_kwargs)
    channel = args.channel
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        start_time = time()
        image = load_image(t, channel)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue

        masks = generator.generate(image)
        labels = convert_annotations_to_labels(masks)

        base = os.path.basename(t)
        savename = os.path.join(args.output, base)

        imwrite(savename, labels, compression = "zlib")
        print(f'Done in {time() - start_time} s')
    print("Done!")


if __name__ == "__main__":
    args = get_args()
    main(args)
