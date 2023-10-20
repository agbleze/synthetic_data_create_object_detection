

#%%
with open("spec_file.txt", mode="w+") as file:
    file.write("enc_key: 'nvidia_tlt'")


# %%
import os

CURRENT_DIR = os.getcwd()
# %%
with open(".env", "w+") as env_file:
    env_file.write(f"CURRENT_DIR={CURRENT_DIR}")

# %%
def create_experiment_spec_file(target_class_map, dataset_config_tfrecords_path,
                                dataset_config_image_directory_path, 
                                inference_config_images_dir,
                                training_config_retrain_pruned_model,
                                inference_config_detection_image_output_dir,
                                inference_config_labels_dump_dir,
                                inference_config_model,
                                
                                )

#%%
with open("default_experiment_spec.cfg", "w+") as spec_file:
    spec_file.writelines(
"""
random_seed: 42
enc_key: 'nvidia_tlt'
verbose: True
model_config {
input_image_config {
image_type: RGB
image_channel_order: 'bgr'
size_height_width {
height: 384
width: 1248
}
    image_channel_mean {
        key: 'b'
        value: 103.939
}
    image_channel_mean {
        key: 'g'
        value: 116.779
}
    image_channel_mean {
        key: 'r'
        value: 123.68
}
image_scaling_factor: 1.0
max_objects_num_per_image: 100
}
arch: "resnet:18"
anchor_box_config {
scale: 64.0
scale: 128.0
scale: 256.0
ratio: 1.0
ratio: 0.5
ratio: 2.0
}
freeze_bn: True
freeze_blocks: 0
freeze_blocks: 1
roi_mini_batch: 256
rpn_stride: 16
use_bias: False
roi_pooling_config {
pool_size: 7
pool_size_2x: False
}
all_projections: True
use_pooling:False
}
dataset_config {
  data_sources: {
    tfrecords_path: "/workspace/tao-experiments/tfrecords/kitti_trainval/kitti_trainval*"
    image_directory_path: "/workspace/tao-experiments/data/training"
  }
image_extension: 'png'
target_class_mapping {
key: 'car'
value: 'car'
}
target_class_mapping {
key: 'van'
value: 'car'
}
target_class_mapping {
key: 'pedestrian'
value: 'person'
}
target_class_mapping {
key: 'person_sitting'
value: 'person'
}
target_class_mapping {
key: 'cyclist'
value: 'cyclist'
}
validation_fold: 0
}
augmentation_config {
preprocessing {
output_image_width: 1248
output_image_height: 384
output_image_channel: 3
min_bbox_width: 1.0
min_bbox_height: 1.0
}
spatial_augmentation {
hflip_probability: 0.5
vflip_probability: 0.0
zoom_min: 1.0
zoom_max: 1.0
translate_max_x: 0
translate_max_y: 0
}
color_augmentation {
hue_rotation_max: 0.0
saturation_shift_max: 0.0
contrast_scale_max: 0.0
contrast_center: 0.5
}
}
training_config {
enable_augmentation: True
enable_qat: False
batch_size_per_gpu: 8
num_epochs: 12
retrain_pruned_model: "/workspace/tao-experiments/data/faster_rcnn/model_1_pruned.tlt"
rpn_min_overlap: 0.3
rpn_max_overlap: 0.7
classifier_min_overlap: 0.0
classifier_max_overlap: 0.5
gt_as_roi: False
std_scaling: 1.0
classifier_regr_std {
key: 'x'
value: 10.0
}
classifier_regr_std {
key: 'y'
value: 10.0
}
classifier_regr_std {
key: 'w'
value: 5.0
}
classifier_regr_std {
key: 'h'
value: 5.0
}

rpn_mini_batch: 256
rpn_pre_nms_top_N: 12000
rpn_nms_max_boxes: 2000
rpn_nms_overlap_threshold: 0.7

regularizer {
type: L2
weight: 1e-4
}

optimizer {
sgd {
lr: 0.02
momentum: 0.9
decay: 0.0
nesterov: False
}
}

learning_rate {
soft_start {
base_lr: 0.02
start_lr: 0.002
soft_start: 0.1
annealing_points: 0.8
annealing_points: 0.9
annealing_divider: 10.0
}
}

lambda_rpn_regr: 1.0
lambda_rpn_class: 1.0
lambda_cls_regr: 1.0
lambda_cls_class: 1.0
}
inference_config {
images_dir: '/workspace/tao-experiments/data/testing/image_2'
model: '/workspace/tao-experiments/data/faster_rcnn/frcnn_kitti_resnet18_retrain.epoch12.tlt'
batch_size: 1
detection_image_output_dir: '/workspace/tao-experiments/data/faster_rcnn/inference_results_imgs_retrain'
labels_dump_dir: '/workspace/tao-experiments/data/faster_rcnn/inference_dump_labels_retrain'
rpn_pre_nms_top_N: 6000
rpn_nms_max_boxes: 300
rpn_nms_overlap_threshold: 0.7
object_confidence_thres: 0.0001
bbox_visualize_threshold: 0.6
classifier_nms_max_boxes: 100
classifier_nms_overlap_threshold: 0.3
}
evaluation_config {
model: '/workspace/tao-experiments/data/faster_rcnn/frcnn_kitti_resnet18_retrain.epoch12.tlt'
batch_size: 1
validation_period_during_training: 1
rpn_pre_nms_top_N: 6000
rpn_nms_max_boxes: 300
rpn_nms_overlap_threshold: 0.7
classifier_nms_max_boxes: 100
classifier_nms_overlap_threshold: 0.3
object_confidence_thres: 0.0001
use_voc07_11point_metric:False
gt_matching_iou_threshold: 0.5
}
                         """)
# %%
