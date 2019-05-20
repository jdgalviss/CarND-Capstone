from styx_msgs.msg import TrafficLight
from object_detection import ObjectDetector
import tfinterface as ti
import cv2
import numpy as np

class TLClassifier(object):
    def __init__(self):
        self.model = ObjectDetector("/capstone/models/mobilenetv1.pb")

        self.traffic_lights_model = ti.estimator.SavedModelPredictor('/capstone/models/traffic-light/v1')

        test_image = np.random.random((200,200,3))
        self.model.predict(image=[test_image])
        self.traffic_lights_model.predict(image=[test_image])
      


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Implement light color prediction
        preds = self.model.predict(image=[image])
        predictions = TrafficLightsUtils.add_traffic_lights_predictions(self.traffic_lights_model, image, preds, normalized_coordinates=True)

        return predictions

class TrafficLightsUtils(object):
    TRAFFIC_CATEGORIES = [ 'Off', 'Red', 'Yellow', 'Green']
    OFFSET = 2

    @classmethod
    def _traffic_lights_predict(self, traffic_lights_model, image, box, normalized_coordinates):
        img_height, img_width, _ = image.shape
        ymin, xmin, ymax, xmax = box

        if normalized_coordinates:
            (xmin, xmax) = (int(xmin * img_width) - self.OFFSET, int(xmax * img_width) + self.OFFSET)
            (ymin, ymax) = (int(ymin * img_height) - self.OFFSET, int(ymax * img_height) + self.OFFSET)
        else:
            (xmin, xmax) = (int(xmin) - self.OFFSET, int(xmax) + self.OFFSET)
            (ymin, ymax) = (int(ymin) - self.OFFSET, int(ymax) + self.OFFSET)

        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        # xmax = xmax if xmax < img_width else img_width - 1 No NEED beause numpy truncates the arrays :O
        # ymax = ymax if ymax < img_height else img_height - 1  
                    
        cropped = image[ymin:ymax + 1, xmin:xmax + 1]
        # cropped = cv2.resize(cropped, (32, 32))

        predictions = traffic_lights_model.predict(image=[cropped])
        label = predictions["classes"][0]
        light = self.TRAFFIC_CATEGORIES[label]
        prob = predictions['probabilities'][0][label]

        return light, prob

    @classmethod
    def add_traffic_lights_predictions(self, traffic_lights_model, image, detections, normalized_coordinates=True):
        for detection in detections:
            category = detection['name']
            if category == 'traffic light':
                light, score = self._traffic_lights_predict(traffic_lights_model, image, detection['box'], normalized_coordinates)
                meta_dict = {"state": light, "estimator": score}
                detection["meta"] = meta_dict
        return detections