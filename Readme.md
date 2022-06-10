# Prerequisites

### Install mmdetection
```
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab
# install latest pytorch prebuilt with the default prebuilt CUDA version (usually the latest)
conda install -c pytorch pytorch torchvision -y
# conda install -c pytorch pytorch=1.5.0 cudatoolkit=10.2 torchvision -y
# install the latest mmcv
pip install mmcv-full --user
# install mmdetection

pip uninstall pycocotools
pip install -r requirements/build.txt
pip install -v -e . --user  # or try "python setup.py develop" if get still got pycocotools error
```

```
conda install scikit-image
```

### prepare dataset
#### NRPerson

To train baseline of NRPerson, you should download all annotations and images here([Google drive](https://drive.google.com/file/d/1xgswDIlPnNTpwF_lKrnUrK5NHtwqjvIE/view?usp=sharing), [Baidu Disk](https://pan.baidu.com/s/193RL4JppDk7XMA5A3-VyoA)  提取码：NRMM)

```
mkdir data
ln -s ${Path_Of_NRMM-FFD} data/NRMM-FFD
```



# Experiment
## NRPerson

```shell script
# exp1: Multi-modal baseline, 8GPU
export GPU=8 && LR=0.04 && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/NRPerson/baseline/faster_rcnn_r50_fpn_1x_NRMM-Person_640_new_filter.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/NRMM-Person/Base/faster_rcnn_r50_fpn_1x_NRMM-Person640/lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}

# exp2 NR-FFD, 8GPU
export GPU=8 && LR=0.04 && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=10000 tools/dist_train.sh configs2/NRPerson/att/faster_rcnn_r50_fpn_1x_NRMM-Person_640_all_filter_paired_att.py $GPU \
  --work-dir ../TOV_mmdetection_cache/work_dir/NRMM-Person/Base/faster_rcnn_r50_fpn_1x_NRMM-Person640/lr${LR}_1x_${GPU}g/ \
  --cfg-options optimizer.lr=${LR}
```

# Examples of Generated Dataset through GTA5
![Example1](https://github.com/NRPerson/NRPerson/blob/main/Generated%20dataset%20Examples/000000.jpg)
![Example2](https://github.com/NRPerson/NRPerson/blob/main/Generated%20dataset%20Examples/000001.jpg)
![Example3](https://github.com/NRPerson/NRPerson/blob/main/Generated%20dataset%20Examples/000007.jpg)
