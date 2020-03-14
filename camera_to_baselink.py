import tf
import rospy
from tf import TransformListener
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np

position = None

def callback(data):
    centroids_in_baselink = []
    if position is not None:
        # Convert each centroid prediction in baselink frame from realsense frame
        #rospy.loginfo("DATA: ", data)
        #rospy.loginfo("POSITION: ", position)
        # data is a 50x4 array, 50 samples of [x,y,z,class_idx]
        data = np.array(data.data).reshape(50,4)
        #print(position)
        for i, row in enumerate(data):
            new_x = row[0] + position[0]
            new_y = row[1] + position[1]
            new_z = row[2] + position[2]
            data[i,:] = np.array([new_x, new_y, new_z, row[3]])
        #    get row to be 
        # publish converted array
        centroid_msg = Float32MultiArray()
        centroid_msg.layout.data_offset = 0
        centroid_msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension()]
        centroid_msg.layout.dim[0].label = "channels"
        centroid_msg.layout.dim[0].size = 4
        centroid_msg.layout.dim[0].stride = 200
        centroid_msg.layout.dim[1].label = "samples"
        centroid_msg.layout.dim[1].size = 50
        centroid_msg.layout.dim[1].stride = 50
        centroid_msg.data = data.flatten()
        pub.publish(centroid_msg)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node('centroid_listener', anonymous = True)
    rospy.Subscriber("object_centroids", Float32MultiArray, callback)
    pub = rospy.Publisher('object_centroids_baselink', Float32MultiArray, queue_size = 10)
    #rospy.init_node("baselink_tf_listener")
    listener = TransformListener()
    rate = rospy.Rate(100.)
    while not rospy.is_shutdown():
        listener.waitForTransform("/base_link", "head_external_camera_depth_frame", rospy.Time(), rospy.Duration(3.0))
        #t = listener.getLatestCommonTime("/base_link", "/head_external_camera_aligned_depth_to_color_frame")
        position, _ = listener.lookupTransform("/base_link", "/head_external_camera_depth_frame", rospy.Time())
        #rate.sleep()
        rospy.spin()
