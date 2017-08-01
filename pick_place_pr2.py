#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point, Quaternion

#import sys
#print(sys.path)

from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
#import message_converter
import yaml


def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

def make_pick_ros(test_num, pick_item, this_object):
    test_scene_num = Int32()
    test_scene_num.data = test_num
    object_name = String()
    object_name.data = pick_item['name']

    pick_pose = Pose()
    pick_pose.position = Point(this_object['pos_x'], this_object['pos_y'], this_object['pos_z'])
    pick_pose.orientation = Quaternion(0.0, 0.0, 0.0, 0.0)

    arm_name = String()
    place_pose = Pose()
    # green tray at the right, [0,-0.71,0.605]
    # red tray at the left, [0,0.71,0.605]
    tray_x = 0.0
    tray_y = 0.71
    tray_z = 0.605 + 0.4

    if pick_item['group'] == 'green':
        arm_name.data = 'right'
        place_pose.position = Point(tray_x, -tray_y, tray_z)
        place_pose.orientation = Quaternion(0.0, 0.0, 0.0, 0.0)
    else:
        arm_name.data = 'left'
        place_pose.position = Point(tray_x, tray_y, tray_z)
        place_pose.orientation = Quaternion(0.0, 0.0, 0.0, 0.0)
    
    return test_scene_num, arm_name, object_name, pick_pose, place_pose

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# pick list yaml file in RoboND-Perception-Project\pr2_robot\config\pick_list_1.yaml
# pick_list_2.yaml, pick_list_3.yaml
def read_pick_list_yaml():
    data_loaded = "Read File Fail"
    with open("../config/pick_list_1.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
    #print(data_loaded)
    return data_loaded['object_list']


# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    # pick_list = read_pick_list_yaml()
    pick_list = rospy.get_param('/object_list')

    # Exercise-2 TODOs:
    print("in pcl_callback")
    # TODO: Convert ROS msg to PCL data
    pcl_data = ros_to_pcl(pcl_msg)

    fil = pcl.StatisticalOutlierRemovalFilter_PointXYZRGB(pcl_data)
    fil.set_mean_k(30)
    fil.set_std_dev_mul_thresh(0.3)
    filtered_data = fil.filter()

    #ros_cloud_filtered = pcl_to_ros(filtered_data)
    #pcl_objects_pub.publish(ros_cloud_filtered)
    #return None

    # TODO: Voxel Grid Downsampling
    vox = filtered_data.make_voxel_grid_filter()
    LEAF_SIZE = 0.01   
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    cloud_filtered = vox.filter()

    # top of table z = 0.63

    # TODO: PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'z'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.6
    axis_max = 1.5
    passthrough.set_filter_limits (axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    passthrough = cloud_filtered.make_passthrough_filter()
    filter_axis = 'x'
    passthrough.set_filter_field_name (filter_axis)
    axis_min = 0.33
    axis_max = 1.5
    passthrough.set_filter_limits (axis_min, axis_max)
    cloud_filtered = passthrough.filter()

    #ros_cloud_filtered = pcl_to_ros(cloud_filtered)
    #pcl_objects_pub.publish(ros_cloud_filtered)
    #return None


    # TODO: RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    # TODO: Extract inliers and outliers
    max_distance = 0.02
    seg.set_distance_threshold(max_distance)
    inliers, coefficients = seg.segment()
    extracted_inliers = cloud_filtered.extract(inliers, negative=False)
    extracted_outliers = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(extracted_outliers) # Apply function to convert XYZRGB to XYZ
    tree = white_cloud.make_kdtree()

    # TODO: Create Cluster-Mask Point Cloud to visualize each cluster separately
    ec = white_cloud.make_EuclideanClusterExtraction()
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(50)
    ec.set_MaxClusterSize(2500)
    ec.set_SearchMethod(tree)
    # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
    cluster_color = get_color_list(len(cluster_indices))

    collision_cloud = cloud_filtered

    # Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    color_cluster_point_list = []
    orig_color_cluster_point_list = []    

    #centroid_list = []
    object_list = []
    min_z = 5

    # Grab the points for the cluster
    for j, indices in enumerate(cluster_indices):
        this_object_point_list = []  
        this_object = {}
        sum_x = 0.0
        sum_y = 0.0
        sum_z = 0.0
        num_pts = 0
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                        rgb_to_float(cluster_color[j])])
            orig_color_cluster_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                        extracted_outliers[indice][3]])
            this_object_point_list.append([white_cloud[indice][0],
                                        white_cloud[indice][1],
                                        white_cloud[indice][2],
                                        extracted_outliers[indice][3]])
            sum_x += white_cloud[indice][0]
            sum_y += white_cloud[indice][1]
            sum_z += white_cloud[indice][2]
            num_pts += 1
            if (min_z > white_cloud[indice][2]):
                min_z = white_cloud[indice][2]


        this_object_cloud = pcl.PointCloud_PointXYZRGB()
        this_object_cloud.from_list(this_object_point_list)

        #print("Min Z",min_z)

        # calculate centroid of each object
        #centroid_list.append([sum_x/num_pts, sum_y/num_pts, sum_z/num_pts])
        this_object['pos_x'] = sum_x/num_pts
        this_object['pos_y'] = sum_y/num_pts
        this_object['pos_z'] = sum_z/num_pts

        # remove object from collision cloud
        #collision_cloud = collision_cloud.extract(indices, negative=True)

        ros_this_object = pcl_to_ros(this_object_cloud)

        # Extract histogram features
        # TODO: complete this step just as you did before in capture_features.py
        chists = compute_color_histograms(ros_this_object, using_hsv=True)
        normals = get_normals(ros_this_object)
        nhists = compute_normal_histograms(normals)
        #nhists = np.random.random(96) 
        
        feature = np.concatenate((chists, nhists))
        #print(feature)
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        this_object['object_name'] = label
        object_list.append(this_object)

        # Publish a label into RViz
        label_pos = list(white_cloud[indices[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, j))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_this_object
        detected_objects.append(do)


    #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    orig_color_cluster_cloud = pcl.PointCloud_PointXYZRGB()
    orig_color_cluster_cloud.from_list(orig_color_cluster_point_list)
    
    cluster_cloud_orig_color = pcl.PointCloud_PointXYZRGB()

    # TODO: Convert PCL data to ROS messages
    ros_cloud_objects = pcl_to_ros(extracted_outliers)
    ros_cloud_table = pcl_to_ros(extracted_inliers)
    ros_cluster_cloud = pcl_to_ros(cluster_cloud)
    ros_collision_cloud = pcl_to_ros(collision_cloud)

    ros_cluster_cloud_orig_color = pcl_to_ros(orig_color_cluster_cloud)

    # TODO: Publish ROS messages
    pcl_objects_pub.publish(ros_cluster_cloud)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_objects_orig_color_pub.publish(ros_cluster_cloud_orig_color)
    collision_map_pub.publish(ros_collision_cloud)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))
    # Publish the list of detected objects
    detected_objects_pub.publish(detected_objects)

    #print("Centroid List")
    #print(centroid_list)

    print("Object List")
    print(object_list)

    print("Objects in Pick List",len(pick_list))
    print("Objects in Detected List",len(object_list))
    num_detected = 0
    for item in pick_list:
        for obj in object_list:
            if item['name'] == obj['object_name']:
                num_detected += 1

    print("Correctly Detected Objects",num_detected)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()
    try:
        pr2_mover()
    except rospy.ROSInterruptException:
        pass


# function to load parameters and request PickPlace service
def pr2_mover():

    # TODO: Initialize variables

    # TODO: Get/Read parameters

    # TODO: Parse parameters into individual variables

    # TODO: Rotate PR2 in place to capture side tables for the collision map
    
    # pi/2 = facing red box in left
    #pub_world_joint.publish(np.pi / 2)

    # TODO: Loop through the pick list

        # TODO: Get the PointCloud for a given object and obtain it's centroid

        # TODO: Create 'place_pose' for the object

        # TODO: Assign the arm to be used for pick_place

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your variables to be sent as a service request
            #resp = pick_place_routine(TEST_SCENE_NUM, OBJECT_NAME, WHICH_ARM, PICK_POSE, PLACE_POSE)

            #print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        # TODO: Output your request parameters into output yaml file

# run_state: Detect -> Scan -> Pick
run_state = 'Detect'


if __name__ == '__main__':

    # TODO: ROS node initialization
    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # TODO: Create Publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)

    pub_world_joint = rospy.Publisher('/pr2/world_joint_controller/command', Float64, queue_size=10)

    pcl_objects_orig_color_pub = rospy.Publisher("/pcl_objects_orig", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_object_array", DetectedObjectsArray, queue_size=1)
    collision_map_pub = rospy.Publisher("/pr2/3d_map/points", PointCloud2, queue_size=1)

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    print("Subscribed..")


    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
