#!/bin/bash
# MobileViT-SSD COCO目标检测训练脚本
# 基于ml-cvnets项目，使用配置文件进行训练

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用的GPU设备
export OMP_NUM_THREADS=8

# COCO数据集路径
COCO_ROOT="/path/to/coco"  # 修改为您的COCO数据集路径
OUTPUT_DIR="./output/mobilevit_ssd_coco"
CONFIG_FILE="config/detection/ssd_coco/mobilevit.yaml"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

echo "========== MobileViT-SSD COCO训练 =========="
echo "配置文件: ${CONFIG_FILE}"
echo "COCO路径: ${COCO_ROOT}"
echo "输出目录: ${OUTPUT_DIR}"
echo "============================================"

# 训练命令
python main_train.py \
    --common.config_file ${CONFIG_FILE} \
    --dataset.root_train ${COCO_ROOT} \
    --dataset.root_val ${COCO_ROOT} \
    --common.results_loc ${OUTPUT_DIR} \
    --dataset.train_batch_size0 32 \
    --dataset.val_batch_size0 32 \
    --dataset.workers 8 \
    --scheduler.max_epochs 200 \
    --optim.adamw.lr 0.0009 \
    --ddp.world_size 4 \
    --common.run_label "mobilevit_ssd_coco"

echo "训练完成!"
