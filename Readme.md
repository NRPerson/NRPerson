# Prerequisites

### [Install mmdetection]
```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
# conda install -c pytorch pytorch=1.5.0 cudatoolkit=10.2 torchvision -y
# install the latest mmcv
pip install mmcv-full --user
# install mmdetection

pip uninstall pycocotools   # sometimes need to source deactivate before, for 
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
```

```
conda install scikit-image
```

- [note]: if your need to modified from origin mmdetection code, see [here](docs/tov/code_modify.md), otherwise do not need any other modified.
- [note]: for more about evaluation, see [evaluation_of_tiny_object.md](docs/tov/evaluation_of_tiny_object.md)
### prepare dataset
#### NRMMPerson

To train baseline of NRMM-Person, you should download all annotation and images.
And we will publish the dataset as soon as possible.


```
mkdir data
ln -s ${Path_Of_NRMM-FFD} data/NRMM-FFD
```



# Experiment
## NRMMPerson

```shell script
# exp1: Multi-modal baseline, 8GPU
export GPU=8 && LR=0.04 && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/NRMM-Person/baseline/faster_rcnn_r50_fpn_1x_NRMM-Person_640_new_filter.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/NRMM-Person/Base/faster_rcnn_r50_fpn_1x_NRMM-Person640/lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# exp2 NRMM-FFD, 8GPU
export GPU=8 && LR=0.04 && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/NRMM-Person/att/faster_rcnn_r50_fpn_1x_NRMM-Person_640_all_filter_paired_att.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/NRMM-Person/Base/faster_rcnn_r50_fpn_1x_NRMM-Person640/lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}
```
