import base64
import os
from io import BytesIO
from PIL import Image
from pathlib import Path
import cv2
import tempfile
import shutil
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
                "images": ("IMAGE", ),
                # "video_path": ("STRING", {"placeholder": "X://path/to/images"}),
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

    def tensor_to_jpeg_folder(self, tensor):
        """
        tensor: torch.Tensor, shape (B, H, W, C), float32 in [0,1]
        returns: str  – absolute path of a *folder* that contains 00000.jpg …
        """
        # 1) pick / create the folder
        base_temp = Path(folder_paths.get_temp_directory())
        base_temp.mkdir(parents=True, exist_ok=True)          # ensure it exists
        tmp_dir = base_temp / f"comfy_vlayout_{os.getpid()}_{id(tensor):x}"

        # 2) wipe & recreate
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 3) dump frames
        tensor_np = (tensor.cpu().numpy() * 255).clip(0, 255).astype(numpy.uint8)
        for idx, frame in enumerate(tensor_np):
            cv2.imwrite(str(tmp_dir / f"{idx:05d}.jpg"),
                        cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        return str(tmp_dir)

    def preprocess(self, images, mode, mask_aug, mask_aug_ratio, bboxes):
        """
        Preprocess the input images based on the selected mode and mask augmentation settings.
        """

        #try:
            # Handle different modes
        if mode == "bboxtrack":
            # processed_frames = self.handle_bboxtrack(video_path, mask_aug, mask_aug_ratio, bboxes)
            jpeg_folder = self.tensor_to_jpeg_folder(images)

            # 1. make sure the folder really exists
            print("DEBUG: temp jpeg folder =", jpeg_folder, os.path.isdir(jpeg_folder))
            # 2. list the files that will be read
            jpgs = sorted(Path(jpeg_folder).glob("*.jpg"))
            print("DEBUG: found", len(jpgs), "frames:", jpgs[:3])
            processed_frames = self.handle_bboxtrack(jpeg_folder, mask_aug, mask_aug_ratio, bboxes)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        #finally:
            # clean-up (ComfyUI temp folder is wiped on restart anyway)
            # shutil.rmtree(jpeg_folder, ignore_errors=True)
        
        return (processed_frames,)

    def handle_bboxtrack(self, jpeg_folder, mask_aug, mask_aug_ratio, bboxes):
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

        input_data = copy.deepcopy(input_params)
        if 'video' in input_params:
            assert jpeg_folder is not None, "Please set video or check configs"
            input_data['video'] = jpeg_folder
            frames, width, height, num_frames = read_video_frames(jpeg_folder, use_type='cv2',  info=True)
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
        pre_save_dir = folder_paths.get_temp_directory()
        if not os.path.exists(pre_save_dir):
            os.makedirs(pre_save_dir)

        # Initialize the annotator and process the frames
        annotator_class = getattr(annotators, class_name)
        annotator_instance = annotator_class(cfg=task_cfg, device=f'cuda:{os.getenv("RANK", 0)}')
        results = annotator_instance.forward(**input_data)

        frames =  results['frames'] if isinstance(results, dict) else results
        print("DEBUG: annotator returned list with len =", len(frames))

        # turn any recognised container into a list of HWC numpy arrays
        if isinstance(frames, numpy.ndarray):
            if frames.ndim == 4:          # (B,H,W,C)
                frames = [frames[i] for i in range(frames.shape[0])]
            elif frames.ndim == 3:        # single image
                frames = [frames]
        elif isinstance(frames, torch.Tensor):
            frames = [f.cpu().numpy() for f in frames.unbind(0)]
        elif isinstance(frames, (list, tuple)):
            frames = [numpy.asarray(f) for f in frames]
        else:
            raise RuntimeError(f"Unexpected type from annotator: {type(frames)}")

        if not frames:
            raise RuntimeError("Annotator returned zero frames.")

        # ---------- float32 tensor in [0,1] ----------
        frames_tensor = torch.stack([
            torch.from_numpy(f).float().to(torch.device('cuda', int(os.getenv("RANK", 0)))) / 255.
            for f in frames
        ])
        return frames_tensor
    
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