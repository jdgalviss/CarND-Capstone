import os
import numpy as np
import tensorflow as tf
import cv2

from collections import Counter
from styx_msgs.msg import TrafficLight


class TLClassifier(object):
    def __init__(self, saved_model_path):
        assert os.path.isdir(saved_model_path), "{} is not a directory".format(saved_model_path)
        self.predictor = tf.contrib.predictor.from_saved_model(saved_model_path)
        self.fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        self.out = cv2.VideoWriter("/output/out.avi", self.fourcc, 10.0, (800,600))


    def get_classification(self, image, threshold=0.4):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        img_array = np.array([image])
        assert len(img_array.shape) == 4, "image should be rank 4, with batch dimmesion on the first dimension"
        raw_predictions = self.predictor({"inputs": img_array})
        predictions = self.parse_predictions(raw_predictions, threshold)
        # Warning: Assuming that we only infer one frame at a time
        assert len(predictions) == 1, "We infered more than one frame."
        prediction = predictions[0]
        detection_classes = raw_predictions['detection_classes'][0][0:4]
        detection_boxes = raw_predictions['detection_boxes'][0][0:4]
        detection_scores = raw_predictions['detection_scores'][0][0:4]
        print(detection_classes)
        print(detection_scores)

        # Write text on the image stating the type of red light
        tl_classes = ['UNKNOWN', 'GREEN','RED', 'YELLOW',  'UNKNOWN']
        message = ""
        if not prediction:
            message = tl_classes[0]
        else:
            classes = [x for x, _ in prediction]
            traffic_light = detection_classes[0]
            if(detection_classes[1] == 2.0 and (detection_scores[1]>=threshold)):
                traffic_light = detection_classes[1]
            message = tl_classes[int(traffic_light)]
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        # Draw bounding boxes
        color=[255, 0, 0]
        thickness=5
        width = 600
        height = 800

        box = detection_boxes[0]
        score = detection_scores[0]
        tl_class = detection_classes[0]
        if(score>threshold):
            if(tl_class == 1.0):
                color = color=[0, 255, 0]
                if(detection_classes[1] == 2.0 and (detection_scores[1]>=threshold)):
                    color = color=[0, 0, 255]
                    box = detection_boxes[1]
            if(tl_class == 2.0):
                color = color=[0, 0, 255]
            if(tl_class == 3.0):
                color = color=[0, 255, 255]
            cv2.line(image, (int(box[1]*height), int(box[0]*width)), (int(box[3]*height), int(box[0]*width)), color, thickness)
            cv2.line(image, (int(box[1]*height), int(box[0]*width)), (int(box[1]*height), int(box[2]*width)), color, thickness)
            cv2.line(image, (int(box[3]*height), int(box[0]*width)), (int(box[3]*height), int(box[2]*width)), color, thickness)
            cv2.line(image, (int(box[1]*height), int(box[2]*width)), (int(box[3]*height), int(box[2]*width)), color, thickness)
            cv2.putText(image,"Traffic Light is: "+message,(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,color,2,cv2.LINE_AA)
    
        self.out.write(image)
        

        if not prediction:
            return TrafficLight.UNKNOWN
        else:
            classes = [x for x, _ in prediction]
            #traffic_light = self.most_frequent(classes)
            traffic_light = detection_classes[0]
            
            if traffic_light == 1.0:
                if(detection_classes[1] == 2.0 and (detection_scores[1]>=threshold)):
                    return TrafficLight.RED
                return TrafficLight.GREEN
            if traffic_light == 2.0:
                return TrafficLight.RED
            if traffic_light == 3.0:
                return TrafficLight.YELLOW

    @staticmethod
    def parse_predictions(predictions, threshold=0.6):
        assert type(predictions) == dict, "Prediction type is {}, not dict".format(type(predictions))
        frames = predictions['num_detections'].shape[0]

        evidence = []
        for frame_index in xrange(frames):
            bool_index = np.greater_equal(predictions['detection_scores'][frame_index], threshold)

            detection_classes = predictions['detection_classes'][frame_index][bool_index]
            detection_scores = predictions['detection_scores'][frame_index][bool_index]
            evidence_frame = zip(detection_classes, detection_scores)

            evidence.append(evidence_frame)

        return evidence

    @staticmethod
    def most_frequent(List): 
        occurence_count = Counter(List) 
        return occurence_count.most_common(1)[0][0]

# from styx_msgs.msg import TrafficLight
# import tensorflow as tf
# import numpy as np
# import datetime

# class TLClassifier(object):
#     def __init__(self, is_sim):

#         if is_sim:
#             PATH_TO_GRAPH = r'/models/frozen_inference_graph.pb'
#         else:
#             PATH_TO_GRAPH = r'/models/frozen_inference_graph.pb'

#         self.graph = tf.Graph()
#         self.threshold = .5

#         with self.graph.as_default():
#             od_graph_def = tf.GraphDef()
#             with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
#                 od_graph_def.ParseFromString(fid.read())
#                 tf.import_graph_def(od_graph_def, name='')

#             self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
#             self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
#             self.scores = self.graph.get_tensor_by_name('detection_scores:0')
#             self.classes = self.graph.get_tensor_by_name('detection_classes:0')
#             self.num_detections = self.graph.get_tensor_by_name(
#                 'num_detections:0')

#         self.sess = tf.Session(graph=self.graph)

#     def get_classification(self, image):
#         """Determines the color of the traffic light in the image
#         Args:
#             image (cv::Mat): image containing the traffic light
#         Returns:
#             int: ID of traffic light color (specified in styx_msgs/TrafficLight)
#         """
#         with self.graph.as_default():
#             img_expand = np.expand_dims(image, axis=0)
#             start = datetime.datetime.now()
#             (boxes, scores, classes, num_detections) = self.sess.run(
#                 [self.boxes, self.scores, self.classes, self.num_detections],
#                 feed_dict={self.image_tensor: img_expand})
#             end = datetime.datetime.now()
#             c = end - start
#             print(c.total_seconds())

#         boxes = np.squeeze(boxes)
#         scores = np.squeeze(scores)
#         classes = np.squeeze(classes).astype(np.int32)

#         print('SCORES: ', scores[0])
#         print('CLASSES: ', classes[0])

#         if scores[0] > self.threshold:
#             if classes[0] == 1:
#                 print('GREEN')
#                 return TrafficLight.GREEN
#             elif classes[0] == 2:
#                 print('RED')
#                 return TrafficLight.RED
#             elif classes[0] == 3:
#                 print('YELLOW')
#                 return TrafficLight.YELLOW

#         return TrafficLight.UNKNOWN