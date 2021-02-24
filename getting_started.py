import pixellib

# What if tensorflow cant get GPU to run? Warnings include cuda, cudart or similar. Do:
# https://ourcodeworld.com/articles/read/1433/how-to-fix-tensorflow-warning-could-not-load-dynamic-library-cudart64-110dll-dlerror-cudart64-110dll-not-found
# BUT! make sure to download a compatible since OpenPose dont support the latest CUDA (tensorflow 2.3+ supports CUDA 11)
# https://stackoverflow.com/a/50622526 -> https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal

# ADDING TO PATH: https://stackoverflow.com/a/60024201
# https://stackoverflow.com/questions/59823283/could-not-load-dynamic-library-cudart64-101-dll-on-tensorflow-cpu-only-install

# Mask RCNN on video
from pixellib.instance import instance_segmentation
INFER_SPEED = 'average' # average or rapid or fast
segment_video = instance_segmentation(infer_speed=INFER_SPEED)
segment_video.load_model("mask_rcnn_coco.h5")
segment_video.process_video("sample.mp4", frames_per_second= 20, output_video_name="output_mask_rcnn.mp4")


# https://pixellib.readthedocs.io/en/latest/video_ade20k.html
from pixellib.semantic import semantic_segmentation
segment_video = semantic_segmentation()
segment_video.load_ade20k_model("deeplabv3_xception65_ade20k.h5")
segment_video.process_video_ade20k("sample.mp4", frames_per_second= 15, output_video_name="output_ade20k.mp4")



