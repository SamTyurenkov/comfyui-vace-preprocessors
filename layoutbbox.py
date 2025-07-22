import base64
import os
from io import BytesIO
from PIL import Image
import numpy
import copy
import folder_paths
import torch
from . import annotators
from .annotators.utils import read_image, read_mask, read_video_frames, save_one_video, save_one_image
from .configs import VACE_PREPROCCESS_CONFIGS

class VideoLayoutTrackAnnotatorNode:
    @classmethod
    def INPUT_TYPES(cls):
        MODES = ["bboxtrack"] # ["masktrack","bboxtrack", "label", "caption"]
        MASKAUG_MODES = ["original", "original_expand", "hull", "hull_expand", "bbox", "bbox_expand"]
        return {
            "required": {
                #"image": ("IMAGE", ),
                "video_path": ("STRING", {"placeholder": "X://path/to/images", "vhs_path_extensions": []}),
                "mode": (MODES, {"default":"bboxtrack"}),
                "mask_aug": (MASKAUG_MODES, {"default":"bbox"}),
                "mask_aug_ratio": ("FLOAT", {"default": 0.1,"min": 0.0, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "bboxes": ("BBOX", {"default": []}),  # Bounding box input as a string (e.g., "x1,y1,x2,y2")
            }
        }
    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "preprocess"

    def preprocess(self, video_path, mode, mask_aug, mask_aug_ratio, bboxes):
        """
        Preprocess the input image based on the selected mode and mask augmentation settings.
        """

        # Handle different modes
        if mode == "bboxtrack":
            processed_frames = self.handle_bboxtrack(video_path, mask_aug, mask_aug_ratio, bboxes)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        return (processed_frames,)

    def handle_bboxtrack(self, video_path, mask_aug, mask_aug_ratio, bboxes):
        """
        Handle bboxtrack mode logic.
        """
        task_cfg = copy.deepcopy(VACE_PREPROCCESS_CONFIGS)["layout_track"]
        print(task_cfg)
        class_name = task_cfg.pop("NAME")
        input_params = task_cfg.pop("INPUTS")
        output_params = task_cfg.pop("OUTPUTS")
        # Placeholder for bboxtrack-specific logic
        print("Processing in bboxtrack mode with mask augmentation:", mask_aug, "and ratio:", mask_aug_ratio)

        fps = None
        input_data = copy.deepcopy(input_params)
        if 'video' in input_params:
            assert video_path is not None, "Please set video or check configs"
            frames, fps, width, height, num_frames = read_video_frames(video_path.split(",")[0], use_type='cv2',  info=True)
            assert frames is not None, "Video read error"
            input_data['frames'] = frames
            input_data['video'] = video_path
        if 'frames' in input_params:
            assert video_path is not None, "Please set video or check configs"
            frames, fps, width, height, num_frames = read_video_frames(video_path.split(",")[0], use_type='cv2', info=True)
            assert frames is not None, "Video read error"
            input_data['frames'] = frames
        if 'bbox' in input_params:
            # assert bbox is not None, "Please set bbox"
            if bboxes is not None:
                input_data['bbox'] = bboxes[0] if len(bboxes) == 1 else bboxes
        if 'mode' in input_params:
            input_data['mode'] = "bboxtrack"
        if 'mask_cfg' in input_params:
            # assert args.maskaug_mode is not None and args.maskaug_ratio is not None, "Please set maskaug_mode and maskaug_ratio or check configs"
            if mask_aug is not None:
                if mask_aug_ratio is not None:
                    input_data['mask_cfg'] = {"mode": mask_aug, "kwargs": {'expand_ratio': mask_aug_ratio, 'expand_iters': 5}}
                else:
                    input_data['mask_cfg'] = {"mode": mask_aug}

        # output data
        save_fps = fps if fps is not None else save_fps
        pre_save_dir = folder_paths.get_temp_directory()
        if not os.path.exists(pre_save_dir):
            os.makedirs(pre_save_dir)

        # Initialize the annotator and process the frames
        annotator_class = getattr(annotators, class_name)
        annotator_instance = annotator_class(cfg=task_cfg, device=f'cuda:{os.getenv("RANK", 0)}')
        results = annotator_instance.forward(**input_data)

        frames =  results['frames'] if isinstance(results, dict) else results
        # Ensure we have a list of individual frames
        if isinstance(frames, numpy.ndarray) and frames.ndim == 4:          # (B, H, W, C)
            frames = [frames[i] for i in range(frames.shape[0])]

        # Convert every frame to float32 in [0, 1] and make it a CUDA tensor
        frames_out = []
        for f in frames:
            if isinstance(f, numpy.ndarray):
                f = torch.from_numpy(f).float()
                if f.device.type == 'cpu':               # not yet on GPU
                    f = f.to(torch.device('cuda', int(os.getenv("RANK", 0))))
                if f.max() > 1.0:                        # scale 0-255 → 0-1
                    f = f / 255.0
            frames_out.append(f)

        # Stack them into one 4-D tensor: (B, H, W, C)
        frames_tensor = torch.stack(frames_out, dim=0)

        if frames_tensor is not None:
            return frames_tensor
        else:
            raise RuntimeError("Annotator did not return any processed frames.")
    
    def parse_bboxes(self, bbox_str):
        """
        Parse bounding box input string into a list of coordinates.
        """
        bboxes = []
        for bbox in bbox_str.split():
            coords = list(map(float, bbox.split(',')))
            if len(coords) != 4:
                raise ValueError(f"The bounding box requires 4 values, but the input is {len(coords)}.")
            bboxes.append(coords)
        return bboxes