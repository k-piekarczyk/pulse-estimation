import numpy as np
import pandas as pd

from typing import Any, Optional, Callable

from pulse_estimation.approach import naive_ICA, naive_PCA, fir_filtered_RG_ICA, fir_filtered_RG_PCA

# face_cascade_file = "./resources/haar/haarcascade_frontalface_default.xml"
face_cascade_file = "./resources/haar/haarcascade_frontalface_alt.xml"


selection = [
    {"video": "./resources/video/my/face_webcam_uncontrolled.mp4", "heart_rate": 80 / 60.0},
    {"video": "./resources/video/my/face_controlled_not_diffused.mov", "heart_rate": 68.5 / 60.0},
    {"video": "./resources/video/my/face_controlled_diffused.mov", "heart_rate": 83.5 / 60.0},
    {"video": "./resources/video/my/face_outdoors_90bpm.mp4", "heart_rate": 90 / 60.0},
    {"video": "./resources/video/other/face.mp4", "heart_rate": 54.0 / 60.0},
]

min_YCrCb = np.array([0, 133, 77], np.uint8)
max_YCrCb = np.array([235, 173, 127], np.uint8)

hr_low = 0.5
hr_high = 3

# naive_ICA(target[0], face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb, acc_hr=target[1])
# naive_PCA(target[0], face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb)

def run_for_every_target(method: Callable[..., Optional[float]], args: tuple, kwargs: dict):
    for target in range(len(selection)):
        target_video = selection[target]["video"]
        target_heart_rate = selection[target].get("heart_rate", None)

        print(f"Current target video: {target_video}")

        result = method(*(target_video, *args), **{"acc_hr": target_heart_rate, **kwargs})

        if result is not None and target_heart_rate is not None:
            target_hr = target_heart_rate * 60
            result_hr = result * 60

            error = abs(result_hr - target_hr)
            error_relative = (error / target_hr) * 100

            print("\t- Target heart rate: %.2f bpm | Estimated heart rate: %.2f bpm" % (target_hr, result_hr))
            print("\t- Error: %.2f bpm (%.2f%%)" % (error, error_relative))
            print("")


def all_targets_ICA():
    run_for_every_target(
        method=fir_filtered_RG_ICA,
        args=(face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb),
        kwargs={"display_face_selection": False, "plot": False},
    )

def all_targets_PCA():
    run_for_every_target(
        method=fir_filtered_RG_PCA,
        args=(face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb),
        kwargs={"display_face_selection": False, "plot": False},
    )


def single_target():
    target = 1
    method=fir_filtered_RG_ICA
    # method = fir_filtered_RG_PCA
    args = (face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb)
    kwargs = {"display_face_selection": False, "plot": False}

    target_video = selection[target]["video"]
    target_heart_rate = selection[target].get("heart_rate", None)

    print(f"Current target video: {target_video}")
    
    result = method(*(target_video, *args), **{"acc_hr": target_heart_rate, **kwargs})

    if result is not None and target_heart_rate is not None:
        target_hr = target_heart_rate * 60
        result_hr = result * 60

        error = abs(result_hr - target_hr)
        error_relative = (error / target_hr) * 100

        print("\t- Target heart rate: %.2f bpm | Estimated heart rate: %.2f bpm" % (target_hr, result_hr))
        print("\t- Error: %.2f bpm (%.2f%%)" % (error, error_relative))
        print("")


def benchmark_single_target():
    iterations = 5
    target = 1
    # method=fir_filtered_RG_ICA
    method = fir_filtered_RG_PCA
    print(method)
    args = (face_cascade_file, hr_low, hr_high, min_YCrCb, max_YCrCb)
    kwargs = {"display_face_selection": False, "plot": False}

    target_video = selection[target]["video"]
    target_heart_rate = selection[target].get("heart_rate", None)

    print(f"Current target video: {target_video}")

    results = []
    for _ in range(iterations):
        result = method(*(target_video, *args), **{"acc_hr": target_heart_rate, **kwargs})

        if result is not None and target_heart_rate is not None:
            target_hr = target_heart_rate * 60
            result_hr = result * 60

            absolute_error = abs(result_hr - target_hr)
            relative_error = (absolute_error / target_hr) * 100
        
        results.append(pd.Series({"est_hr": result_hr, "abs_err": absolute_error, "rel_err": relative_error}))
    print(pd.DataFrame(results))
    
    
