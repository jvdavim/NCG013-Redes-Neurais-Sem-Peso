import cv2
import numpy as np

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
IMG_WIDTH = 416
IMG_HEIGHT = 416


# -------------------------------------------------------------------
# Help functions
# -------------------------------------------------------------------


def get_outputs_names(net: cv2.dnn_Net) -> list:
    """Get the names of the output layers

    Args:
        net (cv2.dnn_Net): Network

    Returns:
        [list]: Names of the output layers, i.e. the layers with unconnected frames
    """
    layers_names = net.getLayerNames()
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def get_face_boxes(frame: np.ndarray, outs: list, conf_threshold: float, nms_threshold: float) -> list:
    """Scan through all the bounding boxes output from the network and keep only
    the ones with high confidence scores. Assign the box's class label as the
    class with the highest score.

    Args:
        frame (np.ndarray): Video frame
        outs (list): Output of yolo network model
        conf_threshold (float): Confidence threshold
        nms_threshold (float): Non maximum suppression threshold

    Returns:
        list: List with frame selected bounds
    """
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)
    return final_boxes


def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom
