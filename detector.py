#!/usr/bin/env python
"""
Wrapper class of YOLOv3 for external usage
"""

from abc import ABC, abstractmethod
from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets import *
from .utils.utils import *
import cv2
import torchvision.transforms as transforms
import rospy
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class Detector(ABC):

    @abstractmethod
    def predict_from_node(self, topic):
        pass

class YOLOv3Detector(Detector):

    """
        YOLOv3 model

        AF(this): an object detector model 
    """

    def __init__(self):
        self.img_size = (320, 320) # (height, width)
        self.half = False
        rospy.init_node('object_detector', anonymous=True)
        self.pub = rospy.Publisher('object_detection', Float32MultiArray, queue_size=10)
        self.rate = rospy.Rate(10) # 10hz
        self.debug = True # For visualization of boxes
        self.image_pub = rospy.Publisher("object_detection_viz", Image, queue_size=10)
        self.bridge = CvBridge()
        
        device = 'cuda'
        # Initialize model
        model = Darknet("/home/joonho1804/catkin_ws/src/uw_common/uw_detection/src/yolov3/cfg/yolov3-spp.cfg", self.img_size)
        weights = '/home/joonho1804/catkin_ws/src/uw_common/uw_detection/src/yolov3/weights/yolov3-spp-ultralytics.pt'
        # attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)
        # model.load_state_dict(torch.load("./weights/yolov3.pt", map_location=device)['model'])

        # Eval mode
        model.to(device).eval()

        # ONNX export mode ignored
        # Half precision
        self.half = self.half and device.type != 'cpu'  # half precision only supported on CUDA
        if (self.half and device == 'cuda'):
            model.half()

        # Get names and colors
        names = load_classes("/home/joonho1804/catkin_ws/src/uw_common/uw_detection/src/yolov3/data/coco.names")
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    def predict_(self, data):
        """
            Callback method for object prediction from image feed
            publishes to topic object_detection
        """
        try:
            # rospy.logfatal("predicting..")
            frame = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
            resized_frame = cv2.resize(frame, (320, 320))
            img = transforms.ToTensor()(resized_frame).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img)[0]
            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    det = det[det[:,4].argsort()]
                    det = det[:50,:]
                    
                    detection_msg = Float32MultiArray()
                    detection_msg.layout.data_offset = 0
                    detection_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
                    detection_msg.layout.dim[0].label = "channels"
                    detection_msg.layout.dim[0].size = 5
                    detection_msg.layout.dim[0].stride = 250
                    detection_msg.layout.dim[1].label = "samples"
                    detection_msg.layout.dim[1].size = 50
                    detection_msg.layout.dim[1].stride = 50
                    detection_msg.data = np.array(detected_centroids).flatten()
                    self.pub.publish(detection_msg)
                
                    if self.debug:
                        # Draw boxes and publish
                        for *xyxy, conf, cls in det:
                            label = '%s %.2f' % (names[int(cls)], conf)
                            plot_one_box(xyxy, img, label=label, color=colors[int(cls)])
                        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                    self.rate.sleep()
        except Exception as e:
            rospy.logfatal(e)
    def predict_from_node(self, topic):
        
        self.sub = rospy.Subscriber(topic, Image, self.predict_)
        rospy.spin()

if __name__== "__main__":
    # detector = YOLOv3Detector()
    # detector.predict_from_node('/head_external_camera/color/image_raw')
    print("test for YOLOv3 Detector")