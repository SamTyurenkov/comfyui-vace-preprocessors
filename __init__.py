from .layoutbbox import VideoLayoutTrackAnnotatorNode
from .layoutbbox import CombineLayoutTracksNode

NODE_CLASS_MAPPINGS = {
    "VideoLayoutTrackAnnotatorNode": VideoLayoutTrackAnnotatorNode,
    "CombineLayoutTracksNode": CombineLayoutTracksNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoLayoutTrackAnnotatorNode": "Video to Layout BBOX Node",
    "CombineLayoutTracksNode": "Combine two BBOX layouts into one",
}