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

StateStore = {}

# EDIT this to match the test world number
StateStore['test_num'] = 1

# run_state: Detect -> ScanLeft -> ScanRight -> GoCenter -> Pick
StateStore['run_state'] = 'Detect'

# run_mode: Normal, SkipScan
StateStore['run_mode'] = 'SkipScan'

# Number of times we run the detection
StateStore['detection_count'] = 0
# Max number of times we run the detection
StateStore['max_detection_count'] = 10

StateStore['scan_left_count'] = 0
StateStore['max_scan_left_count'] = 7

StateStore['scan_right_count'] = 0
StateStore['max_scan_right_count'] = 14

StateStore['go_center_count'] = 0
StateStore['max_go_center_count'] = 7

# collision_cloud as global variable
StateStore['collision_cloud'] = pcl.PointCloud_PointXYZRGB()


StateStore['detected_objects_labels'] = []
StateStore['detected_objects'] = []

StateStore['pick_list'] = []
StateStore['object_list'] = []

StateStore['pick_ros_list'] = []
StateStore['label_marker_list'] = []

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
    tray_z = 0.605 + 0.1

    if pick_item['group'] == 'green':
        arm_name.data = 'right'
        place_pose.position = Point(tray_x, -tray_y, tray_z)
        place_pose.orientation = Quaternion(0.0, 0.0, 0.0, 0.0)
    else:
        arm_name.data = 'left'
        place_pose.position = Point(tray_x, tray_y, tray_z)
        place_pose.orientation = Quaternion(0.0, 0.0, 0.0, 0.0)
    
    return test_scene_num, arm_name, object_name, pick_pose, place_pose

def generate_full_ros_pick():
    global StateStore
    pick_ros = []
    pick_yaml = []

    test_num = StateStore['test_num']
    for item in StateStore['pick_list']:
        for obj in StateStore['object_list']:
            if item['name'] == obj['object_name']:
                test_scene_num, arm_name, object_name, pick_pose, place_pose = make_pick_ros(test_num, item, obj)
                pick_ros.append([test_scene_num, arm_name, object_name, pick_pose, place_pose])
                yaml = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                pick_yaml.append(yaml)

    send_to_yaml("output_" + str(test_num) + ".yaml", pick_yaml)
    StateStore['pick_ros_list'] = pick_ros
    #print(pick_ros)

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

# add append_cloud points into main_cloud
def merge_pcl_cloud(main_cloud, append_cloud):
    XYZRGB_cloud = pcl.PointCloud_PointXYZRGB()
    points_list = []

    float_rgb = rgb_to_float([200,0,200])
    x_min = 0.33
    z_min = 0.62

    for data in main_cloud:
        # ignore the objects
        if (data[0] > x_min) and (data[2] > z_min):
            abc = 1
        else:
            points_list.append([data[0], data[1], data[2], float_rgb])
    for data in append_cloud:
        if (data[0] > x_min) and (data[2] > z_min):
            abc = 1
        else:
            points_list.append([data[0], data[1], data[2], float_rgb])

    XYZRGB_cloud.from_list(points_list)

    # Voxel Grid Downsampling to remove repeated points
    vox = XYZRGB_cloud.make_voxel_grid_filter()
    LEAF_SIZE = 0.01   
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
    XYZRGB_cloud = vox.filter()

    return XYZRGB_cloud

# Function for Object Detection
def object_detection(cloud_filtered):
    global StateStore
    StateStore['detection_count'] += 1
    print(StateStore['run_state'],StateStore['detection_count'],StateStore['max_detection_count'])

    # pick_list = read_pick_list_yaml()
    pick_list = rospy.get_param('/object_list')

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

    collision_cloud = cloud_filtered

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


    # Exercise-3 TODOs:

    # Classify the clusters! (loop through each detected cluster one at a time)
    detected_objects_labels = []
    detected_objects = []

    color_cluster_point_list = []
    orig_color_cluster_point_list = []    

    #centroid_list = []
    object_list = []
    min_z = 5
    label_list = []

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
        label_list.append([label,label_pos, j])

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

    #print("Object List")
    #print(object_list)

    print("Objects in Pick List",len(pick_list))
    print("Objects in Detected List",len(object_list))
    num_detected = 0
    for item in pick_list:
        for obj in object_list:
            if item['name'] == obj['object_name']:
                num_detected += 1

    print("Correctly Detected Objects",num_detected)

    # go to next state if detection complete
    if (num_detected >= len(pick_list)) or (StateStore['detection_count'] >= StateStore['max_detection_count']):
        # save into global variable
        StateStore['detected_objects_labels'] = detected_objects_labels
        StateStore['detected_objects'] = detected_objects
        StateStore['pick_list'] = pick_list
        StateStore['object_list'] = object_list
        StateStore['label_marker_list'] = label_list
        generate_full_ros_pick()

        # go to next step
        if StateStore['run_mode'] == 'SkipScan':
            StateStore['run_state'] = 'LoadCollisionMap'
        else:
            StateStore['run_state'] = 'ScanLeft'
            pub_world_joint.publish(np.pi / 2)


def scan_left(cloud_filtered):
    global StateStore
    StateStore['scan_left_count'] += 1
    print(StateStore['run_state'],StateStore['scan_left_count'],StateStore['max_scan_left_count'])

    StateStore['collision_cloud'] = merge_pcl_cloud(StateStore['collision_cloud'], cloud_filtered)

    # go to next state if Scan Left complete
    if (StateStore['scan_left_count'] >= StateStore['max_scan_left_count']):
        StateStore['run_state'] = 'ScanRight'
        pub_world_joint.publish(-np.pi / 2)

def scan_right(cloud_filtered):
    global StateStore
    StateStore['scan_right_count'] += 1
    print(StateStore['run_state'],StateStore['scan_right_count'],StateStore['max_scan_right_count'])

    StateStore['collision_cloud'] = merge_pcl_cloud(StateStore['collision_cloud'], cloud_filtered)

    # go to next state if Scan Left complete
    if (StateStore['scan_right_count'] >= StateStore['max_scan_right_count']):
        StateStore['run_state'] = 'GoCenter'
        # Save collision cloud to disk
        pickle.dump(StateStore['collision_cloud'], open('pr2_collision_cloud.sav', 'wb'))
        pub_world_joint.publish(0.0)

def go_center(cloud_filtered):
    global StateStore
    StateStore['go_center_count'] += 1
    print(StateStore['run_state'],StateStore['go_center_count'],StateStore['max_go_center_count'])

    # go to next state if Scan Left complete
    if (StateStore['go_center_count'] >= StateStore['max_go_center_count']):
        StateStore['run_state'] = 'Pick'



# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

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

    global StateStore
    if StateStore['run_state'] == 'Detect':
        object_detection(cloud_filtered)
    elif StateStore['run_state'] == 'ScanLeft':
        scan_left(cloud_filtered)
    elif StateStore['run_state'] == 'ScanRight':
        scan_right(cloud_filtered)
    elif StateStore['run_state'] == 'GoCenter':
        go_center(cloud_filtered)
    elif StateStore['run_state'] == 'LoadCollisionMap':
        load_collision_map(cloud_filtered)
    elif StateStore['run_state'] == 'Pick':
        start_pick(cloud_filtered)
    elif StateStore['run_state'] == 'Done':
        print(StateStore['run_state'])

    ros_collision_cloud = pcl_to_ros(StateStore['collision_cloud'])
    collision_map_pub.publish(ros_collision_cloud)

    for thislabel in StateStore['label_marker_list']:
        object_markers_pub.publish(make_label(thislabel[0],thislabel[1], thislabel[2]))


def load_collision_map(cloud_filtered):
    global StateStore
    # Load collision cloud from disk
    StateStore['collision_cloud'] = pickle.load(open('pr2_collision_cloud.sav', 'rb'))
    StateStore['run_state'] = 'Pick'


# function to load parameters and request PickPlace service
def start_pick(cloud_filtered):
    global StateStore
    # pi/2 = facing red box in left
    #pub_world_joint.publish(np.pi / 2)

    # TODO: Loop through the pick list
    for item in StateStore['pick_ros_list']:
        # TODO: Get the PointCloud for a given object and obtain it's centroid
        test_scene_num = item[0]
        arm_name = item[1]
        object_name = item[2]
        pick_pose = item[3]
        place_pose = item[4]

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

            # TODO: Insert your variables to be sent as a service request
            print ("Start Picking: ",test_scene_num, object_name, arm_name)
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

            print ("Response: ",resp.success)

        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        # TODO: Output your request parameters into output yaml file
    StateStore['run_state'] = 'Done'

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
    #pub_world_joint.publish(0.0)


    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
