" script of detection models that encapsulates most of the object detection architectures "
from collections import namedtuple
import numpy as np
from phi import P, Obj, Rec
import cytoolz as cz
import tfinterface as ti


def objectdetectionprediction():
    return namedtuple('ObjectDetectionPrediction', ['name', 'score', 'box'])

class ObjectDetector(ti.estimator.FrozenGraphPredictor):
    """Detector class for general models for object detection"""
    def __init__(self, frozen_graph_path, *args, **kargs):
        input_nodes = dict(image = "image_tensor:0")
        output_names = dict(num_detections = "num_detections:0",boxes  =  "detection_boxes:0", scores =  "detection_scores:0",classes = "detection_classes:0")
    
        super(ObjectDetector, self).__init__(frozen_graph_path, input_nodes, output_names, *args, **kargs)
    
    def predict(self, **kargs):
        min_score = kargs.pop("min_score", 0.5)
        max_predictions = kargs.pop("max_predictions", 20)

        predictions = super(ObjectDetector, self).predict(**kargs)
        predictions['num_detections'] = int(predictions['num_detections'][0].tolist())
        predictions['classes'] = predictions[
            'classes'][0].astype(np.uint8).tolist()
        predictions['boxes'] = predictions['boxes'][0].tolist()
        predictions['scores'] = predictions['scores'][0].tolist()

        predictions = zip(predictions['classes'], predictions['scores'], predictions['boxes'])
        predictions = map(create_pred, predictions)
        predictions = filter(P["score"] > min_score, predictions)
        predictions = sorted(predictions, key = P["score"])
        predictions = cz.take(max_predictions, predictions)
        predictions = list(predictions)

        return predictions

def create_pred(tup):
    name, score, box = tup
    name = CATEGORY_INDEX[name]
    distance = distance_approximator(box)
    return dict(name=name["name"], score=score, box=box, id=name['id'], distance=distance)

def distance_approximator(box):
    ymin, xmin, ymax, xmax = box
    mid_x = (xmax + xmin) / 2
    mid_y = (ymax + ymin) / 2  # TODO: use mid_y
    apx_distance = round((1 - (xmax - xmin)) ** 4, 1)
    return apx_distance

def detection_list(boxes, classes, scores, max_boxes_to_report=None, min_score_thresh=.5):
    """Create dictionary of detections"""
    detections = list()
    if not max_boxes_to_report:
        max_boxes_to_report = len(boxes)
    for i in range(min(max_boxes_to_report, len(boxes))):
        if scores is None or scores[i] > min_score_thresh:

            class_name = CATEGORY_INDEX[classes[i]]['name']
            box = tuple(boxes[i].tolist())
            score = int(100*scores[i])
            detections.append((class_name, {"score": score, "coordinates": box}))
    return detections


CATEGORY_INDEX = {
    1: {'id': 1, 'name': 'person'},
    2: {'id': 2, 'name': 'bicycle'},
    3: {'id': 3, 'name': 'car'},
    4: {'id': 4, 'name': 'motorcycle'},
    5: {'id': 5, 'name': 'airplane'},
    6: {'id': 6, 'name': 'bus'},
    7: {'id': 7, 'name': 'train'},
    8: {'id': 8, 'name': 'truck'},
    9: {'id': 9, 'name': 'boat'},
    10: {'id': 10, 'name': 'traffic light'},
    11: {'id': 11, 'name': 'fire hydrant'},
    13: {'id': 13, 'name': 'stop sign'},
    14: {'id': 14, 'name': 'parking meter'},
    15: {'id': 15, 'name': 'bench'},
    16: {'id': 16, 'name': 'bird'},
    17: {'id': 17, 'name': 'cat'},
    18: {'id': 18, 'name': 'dog'},
    19: {'id': 19, 'name': 'horse'},
    20: {'id': 20, 'name': 'sheep'},
    21: {'id': 21, 'name': 'cow'},
    22: {'id': 22, 'name': 'elephant'},
    23: {'id': 23, 'name': 'bear'},
    24: {'id': 24, 'name': 'zebra'},
    25: {'id': 25, 'name': 'giraffe'},
    27: {'id': 27, 'name': 'backpack'},
    28: {'id': 28, 'name': 'umbrella'},
    31: {'id': 31, 'name': 'handbag'},
    32: {'id': 32, 'name': 'tie'},
    33: {'id': 33, 'name': 'suitcase'},
    34: {'id': 34, 'name': 'frisbee'},
    35: {'id': 35, 'name': 'skis'},
    36: {'id': 36, 'name': 'snowboard'},
    37: {'id': 37, 'name': 'sports ball'},
    38: {'id': 38, 'name': 'kite'},
    39: {'id': 39, 'name': 'baseball bat'},
    40: {'id': 40, 'name': 'baseball glove'},
    41: {'id': 41, 'name': 'skateboard'},
    42: {'id': 42, 'name': 'surfboard'},
    43: {'id': 43, 'name': 'tennis racket'},
    44: {'id': 44, 'name': 'bottle'},
    46: {'id': 46, 'name': 'wine glass'},
    47: {'id': 47, 'name': 'cup'},
    48: {'id': 48, 'name': 'fork'},
    49: {'id': 49, 'name': 'knife'},
    50: {'id': 50, 'name': 'spoon'},
    51: {'id': 51, 'name': 'bowl'},
    52: {'id': 52, 'name': 'banana'},
    53: {'id': 53, 'name': 'apple'},
    54: {'id': 54, 'name': 'sandwich'},
    55: {'id': 55, 'name': 'orange'},
    56: {'id': 56, 'name': 'broccoli'},
    57: {'id': 57, 'name': 'carrot'},
    58: {'id': 58, 'name': 'hot dog'},
    59: {'id': 59, 'name': 'pizza'},
    60: {'id': 60, 'name': 'donut'},
    61: {'id': 61, 'name': 'cake'},
    62: {'id': 62, 'name': 'chair'},
    63: {'id': 63, 'name': 'couch'},
    64: {'id': 64, 'name': 'potted plant'},
    65: {'id': 65, 'name': 'bed'},
    67: {'id': 67, 'name': 'dining table'},
    70: {'id': 70, 'name': 'toilet'},
    72: {'id': 72, 'name': 'tv'},
    73: {'id': 73, 'name': 'laptop'},
    74: {'id': 74, 'name': 'mouse'},
    75: {'id': 75, 'name': 'remote'},
    76: {'id': 76, 'name': 'keyboard'},
    77: {'id': 77, 'name': 'cell phone'},
    78: {'id': 78, 'name': 'microwave'},
    79: {'id': 79, 'name': 'oven'},
    80: {'id': 80, 'name': 'toaster'},
    81: {'id': 81, 'name': 'sink'},
    82: {'id': 82, 'name': 'refrigerator'},
    84: {'id': 84, 'name': 'book'},
    85: {'id': 85, 'name': 'clock'},
    86: {'id': 86, 'name': 'vase'},
    87: {'id': 87, 'name': 'scissors'},
    88: {'id': 88, 'name': 'teddy bear'},
    89: {'id': 89, 'name': 'hair drier'},
    90: {'id': 90, 'name': 'toothbrush'}
    }
