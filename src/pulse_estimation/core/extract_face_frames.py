import cv2
import numpy as np
import numpy.typing as npt

from typing import Optional

from .exceptions import NoFaceDetected
from pulse_estimation.utils.video import FileVideoSource


__all__ = ["extract_face_frames"]


def extract_face_frames(vid: FileVideoSource, face_cascade_path: str, display: bool = False) -> npt.NDArray[np.uint8]:
    """
    Extracts the face from the provided `FileVideoSource`.
    This closes the `FileVideoSource`.
    """
    face_frames = []

    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    cap = vid.get_cap()

    face_detected = False
    while True:
        ret, frame = cap.read()

        if ret:
            face: Optional[cv2.Mat] = None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                face_detected = True
                x, y, w, h = faces[0]
                face = frame[y : y + h, x : x + w].copy()
                face_frames.append(face)

                if display:
                    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.imshow("Original frame", frame)
                    cv2.imshow("Extracted face", face)
            elif face_detected:
                break

            if display and (cv2.waitKey(1) & 0xFF == ord("q")):
                break
        else:
            break

    if display:
        cap.release()
        cv2.destroyAllWindows()

    if len(face_frames) == 0:
        raise NoFaceDetected()

    return np.array(face_frames, dtype=np.ndarray)
