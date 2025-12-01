# risk_detection/utils/pose_utils.py
import numpy as np

def iter_keypoints(res_pose):
    if not res_pose or not hasattr(res_pose[0], "keypoints") or res_pose[0].keypoints is None:
        return []
    return res_pose[0].keypoints.xy.cpu().numpy()  # [N,17,2]

def iter_feet(res_pose, feet_idxs=(15,16)):
    kps = iter_keypoints(res_pose)
    for kp_set in kps:
        for idx in feet_idxs:
            x, y = kp_set[idx]
            yield float(x), float(y)
