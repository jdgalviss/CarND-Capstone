import os
import numpy as np
import tensorflow as tf
import cv2

from collections import Counter
from styx_msgs.msg import TrafficLight

MODEL_PATH_REAL = r'light_classification/models/mobilenetv1_real_2/saved_model'
MODEL_PATH_SIM = r'light_classification/models/mobilenetv1_2/saved_model'
class TLClassifier(object):
    def __init__(self, is_sim):
        self.is_model_ready = False
        if is_sim:
            MODEL_PATH = MODEL_PATH_SIM
            self.fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            self.out = cv2.VideoWriter("/output/out.avi", self.fourcc, 10.0, (800,600))
        else:
            MODEL_PATH = MODEL_PATH_REAL
            # self.fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            # self.out = cv2.VideoWriter("/output/out.avi", self.fourcc, 10.0, (800,600))
        assert os.path.isdir(MODEL_PATH), "{} is not a directory".format(MODEL_PATH)
        self.predictor = tf.contrib.predictor.from_saved_model(MODEL_PATH)
        self.is_sim = is_sim
        print("Testing Traffic light detection model on black image")
        test_prediction = self.predictor({"inputs": [np.zeros((800,600,3))]})
        print("Traffic light detection model ready!")
        self.is_model_ready = True
        


    def get_classification(self, image, threshold=0.15):
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
        #print(detection_classes)
        #print(detection_scores)

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

        if self.is_sim:
            for box, score, tl_class in zip(detection_boxes, detection_scores, detection_classes):
                if(score>threshold):
                    if(tl_class == 1.0):
                        color = color=[0, 255, 0]
                    if(tl_class == 2.0):
                        color = color=[0, 0, 255]
                    if(tl_class == 3.0):
                        color = color=[0, 255, 255]
                    cv2.line(image, (int(box[1]*height), int(box[0]*width)), (int(box[3]*height), int(box[0]*width)), color, thickness)
                    cv2.line(image, (int(box[1]*height), int(box[0]*width)), (int(box[1]*height), int(box[2]*width)), color, thickness)
                    cv2.line(image, (int(box[3]*height), int(box[0]*width)), (int(box[3]*height), int(box[2]*width)), color, thickness)
                    cv2.line(image, (int(box[1]*height), int(box[2]*width)), (int(box[3]*height), int(box[2]*width)), color, thickness)
                    cv2.putText(image,"Traffic Light is: "+message,(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,color,2,cv2.LINE_AA)
                    #self.out.write(image)
        # else:
        #     box = detection_boxes[0]
        #     score = detection_scores[0]
        #     tl_class = detection_classes[0]
        #     if(score>threshold):
        #         if(tl_class == 1.0):
        #             color = color=[0, 255, 0]
        #             if(detection_classes[1] == 2.0 and (detection_scores[1]>=threshold)):
        #                 color = color=[0, 0, 255]
        #                 box = detection_boxes[1]
        #         if(tl_class == 2.0):
        #             color = color=[0, 0, 255]
        #         if(tl_class == 3.0):
        #             color = color=[0, 255, 255]
        #         cv2.line(image, (int(box[1]*height), int(box[0]*width)), (int(box[3]*height), int(box[0]*width)), color, thickness)
        #         cv2.line(image, (int(box[1]*height), int(box[0]*width)), (int(box[1]*height), int(box[2]*width)), color, thickness)
        #         cv2.line(image, (int(box[3]*height), int(box[0]*width)), (int(box[3]*height), int(box[2]*width)), color, thickness)
        #         cv2.line(image, (int(box[1]*height), int(box[2]*width)), (int(box[3]*height), int(box[2]*width)), color, thickness)
        #         cv2.putText(image,"Traffic Light is: "+message,(20,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5,color,2,cv2.LINE_AA)
        
        

        if not prediction:
            return TrafficLight.UNKNOWN
        else:
            classes = [x for x, _ in prediction]
            #traffic_light = self.most_frequent(classes)
            traffic_light = detection_classes[0]
            
            if traffic_light == 1.0:
                #if(detection_classes[1] == 2.0 and (detection_scores[1]>=threshold)):
                #    return TrafficLight.RED
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