# Udacity Robotics Project 3 - 3D Perception

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

The trained model achieve an accuracy of 95%.
