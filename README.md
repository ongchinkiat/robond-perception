# Udacity Robotics Project 3 - 3D Perception

The project python file is in pick_place_pr2.py


## Exercise 1: Pipeline for filtering and RANSAC plane fitting

In this project, the input point cloud received from the camera is noisy.

The first step in the pipeline is to filter out some of the noise, using the Statistical Outlier Removal Filter.

```
fil = pcl.StatisticalOutlierRemovalFilter_PointXYZRGB(pcl_data)
fil.set_mean_k(30)
fil.set_std_dev_mul_thresh(0.3)
filtered_data = fil.filter()
```

Next, I cut down the sample size to make processing faster.

```
vox = filtered_data.make_voxel_grid_filter()
LEAF_SIZE = 0.01   
vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
cloud_filtered = vox.filter()
```

The next step is to filter out the floor, so that only the table and objects remain.

```
passthrough = cloud_filtered.make_passthrough_filter()
filter_axis = 'z'
passthrough.set_filter_field_name (filter_axis)
axis_min = 0.6
axis_max = 1.5
passthrough.set_filter_limits (axis_min, axis_max)
cloud_filtered = passthrough.filter()
```

The camera view contains the edges of the Drop Boxes. I used another filter on the x axis to filter them out.

```
passthrough = cloud_filtered.make_passthrough_filter()
filter_axis = 'x'
passthrough.set_filter_field_name (filter_axis)
axis_min = 0.33
axis_max = 1.5
passthrough.set_filter_limits (axis_min, axis_max)
cloud_filtered = passthrough.filter()
```

Then, a RANSAC plane fitting is used to seperate the objects and the table.

```
seg = cloud_filtered.make_segmenter()
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)

max_distance = 0.02
seg.set_distance_threshold(max_distance)
inliers, coefficients = seg.segment()
extracted_inliers = cloud_filtered.extract(inliers, negative=False)
extracted_outliers = cloud_filtered.extract(inliers, negative=True)
```

After the RANSAC plane fitting, the table is in extracted_inliers, and the objects are in extracted_outliers.

## Exercise 2: Pipeline for cluster segmentation

The object point cloud is used for cluster segmentation, using the Euclidean Cluster Extraction method.

```
ec = white_cloud.make_EuclideanClusterExtraction()
ec.set_ClusterTolerance(0.03)
ec.set_MinClusterSize(50)
ec.set_MaxClusterSize(2500)
ec.set_SearchMethod(tree)
cluster_indices = ec.Extract()
```

## Exercise 3: Object recognition

For object recognition, the object color and normal histograms are computed.

For each object model, 50 sample instances are used. The models are trained using SVM using a Linear Kernel.

The trained model achieve an accuracy of 95%.

![Confusion Matrix](https://github.com/ongchinkiat/robond-perception/raw/master/pr2_train_matrix.jpg "Confusion Matrix")

In the perception pipeline, the histogram features are computed for each object, and prediction is done using the trained SVM model.

```
chists = compute_color_histograms(ros_this_object, using_hsv=True)
normals = get_normals(ros_this_object)
nhists = compute_normal_histograms(normals)

feature = np.concatenate((chists, nhists))
clf.predict(scaler.transform(feature.reshape(1,-1)))
label = encoder.inverse_transform(prediction)[0]
```

## Pick and Place Setup

The 3 test worlds are used to test out the object recognition model. The outputs are saved in output_1.yaml, output_2.yaml, output_3.yaml.

The pipeline achieve 100% recognition in all 3 test worlds.

World 1:
![World 1](https://github.com/ongchinkiat/robond-perception/raw/master/world1.jpg "World 1")

World 2:
![World 2](https://github.com/ongchinkiat/robond-perception/raw/master/world2.jpg "World 2")

World 3:
![World 3](https://github.com/ongchinkiat/robond-perception/raw/master/world3.jpg "World 3")

To make the PR2 robot complete the pick and place tasks, a simple state machine was added to the pick_place_pr2.py script. The states are: Detect -> ScanLeft -> ScanRight -> GoCenter -> Pick.

I have uploaded a video of the simulator running all these tasks for World 1.

Video URL: https://youtu.be/SIkD2-r9jj4

<a href="http://www.youtube.com/watch?feature=player_embedded&v=SIkD2-r9jj4" target="_blank"><img src="http://img.youtube.com/vi/SIkD2-r9jj4/0.jpg"
alt="PR2 Pick and Place" width="240" height="180" border="1" /></a>


In the first state, Detect, the whole recognition pipeline is run to detect the objects. Since the input is noisy, a few detection rounds may be needed.

In the next 2 state, ScanLeft and ScanRight, we sent commands to the World Joint of the robot to make it turn left, then right. While the robot is turning, a collision map generation pipeline is used to map out the table and the drop box.

Collision Map Generation:
![Collision Map](https://github.com/ongchinkiat/robond-perception/raw/master/collisionmap.jpg "Collision Map")

Once the complete collision map is generated, we make the robot go back to the center position, in the GoCenter state.

Finally, in the Pick state, we go through the recognised object list and send the pick and place command using the pick_place_routine.
