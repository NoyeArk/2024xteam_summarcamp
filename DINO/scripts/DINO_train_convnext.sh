coco_path="D:/Code/Paper-code/DINO/ExDark/data"
backbone_dir="D:/Code/Paper-code/DINO/models/dino"

#export CUDA_VISIBLE_DEVICES=$3 && python main.py \
#	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale_convnext.py --coco_path $coco_path \
#	--options dn_scalar=100 embed_init_tgt=TRUE \
#	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
#	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir

export CUDA_VISIBLE_DEVICES=$3 && python main.py \
	--output_dir logs/DINO/R50-MS4 -c config/DINO/DINO_4scale_convnext.py --coco_path $coco_path \
	--dataset_file=coco --options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0 backbone_dir=$backbone_dir