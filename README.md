# MM-OR: A Large Multimodal Operating Room Dataset for Semantic Understanding of High-Intensity Surgical Environments

<img align="right" src="figure.jpg" alt="teaser" width="100%" style="margin-left: 10px">

Official code of the paper "MM-OR: A Large Multimodal Operating Room Dataset for Semantic Understanding of High-Intensity Surgical Environments" accepted at CVPR 2025. Operating rooms (ORs) are complex, high-stakes environments requiring precise understanding of interactions among medical staff, tools, and equipment for enhancing surgical assistance, situational awareness, and patient safety. Current datasets fall short in scale, realism and do not capture the multimodal nature of OR scenes, limiting progress in OR modeling. To this end, we introduce MM-OR, a realistic and large-scale multimodal spatiotemporal OR dataset, and the first dataset to enable multimodal scene graph generation. MM-OR captures comprehensive OR scenes containing RGB-D data, detail views, audio, speech transcripts, robotic logs, and tracking data and is annotated with panoptic segmentations, semantic scene graphs, and downstream task labels. Further, we propose MM2SG, the first multimodal large vision-language model for scene graph generation, and through extensive experiments, demonstrate its ability to effectively leverage multimodal inputs. Together, MM-OR and MM2SG establish a new benchmark for holistic OR understanding, and open the path towards multimodal scene analysis in complex, high-stakes environments. More details are provided in the paper.

TODO
Paper: To be added after it is published

**Authors**: [Ege Özsoy][eo], Chantal Pellegrini, Tobias Czempiel, Felix Tristram, Kun Yuan, David Bani-Harouni, Ulrich Eck, Benjamin Busam, Matthias Keicher, [Nassir Navab][nassir]

[eo]: https://www.cs.cit.tum.de/camp/members/ege-oezsoy/
[nassir]: https://www.cs.cit.tum.de/camp/members/cv-nassir-navab/nassir-navab/

```
@inproceedings{ozsoy2024mmor,
  title={MM-OR: A Large Multimodal Operating Room Dataset for Semantic Understanding of High Intensity Surgical Environments},
  author={\textbf{Ege Özsoy} and Pellegrini, Chantal and Czempiel, Tobias and Tristram, Felix and Yuan, Kun and Bani-Harouni, David and Eck, Ulrich and Busam, Benjamin and Keicher, Matthias and Navab, Nassir},
  booktitle={CVPR},
  note={Accepted},
  year={2025}
}

@inproceedings{ozsoy2024oracle,
  title={ORacle: Large Vision-Language Models for Knowledge-Guided Holistic OR Domain Modeling},
  author={{\"O}zsoy, Ege and Pellegrini, Chantal and Keicher, Matthias and Navab, Nassir},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={455--465},
  year={2024},
  organization={Springer}
}

@inproceedings{Özsoy2023_LABRAD_OR,
    title={LABRAD-OR: Lightweight Memory Scene Graphs for Accurate Bimodal Reasoning in Dynamic Operating Rooms},
    author={Ege Özsoy, Tobias Czempiel, Felix Holm, Chantal Pellegrini, Nassir Navab},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    year={2023},
    organization={Springer}
}
@Article{Özsoy2023,
author={{\"O}zsoy, Ege
and Czempiel, Tobias
and {\"O}rnek, Evin P{\i}nar
and Eck, Ulrich
and Tombari, Federico
and Navab, Nassir},
title={Holistic OR domain modeling: a semantic scene graph approach},
journal={International Journal of Computer Assisted Radiology and Surgery},
year={2023},
doi={10.1007/s11548-023-03022-w},
url={https://doi.org/10.1007/s11548-023-03022-w}
}
@inproceedings{Özsoy2022_4D_OR,
    title={4D-OR: Semantic Scene Graphs for OR Domain Modeling},
    author={Ege Özsoy, Evin Pınar Örnek, Ulrich Eck, Tobias Czempiel, Federico Tombari, Nassir Navab},
    booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
    year={2022},
    organization={Springer}
}
@inproceedings{Özsoy2021_MSSG,
    title={Multimodal Semantic Scene Graphs for Holistic Modeling of Surgical Procedures},
    author={Ege Özsoy, Evin Pınar Örnek, Ulrich Eck, Federico Tombari, Nassir Navab},
    booktitle={Arxiv},
    year={2021}
}
```

## MM-OR Dataset
- To download MM-OR, first fill out this form https://forms.gle/kj47QXEcraQdGidg6 to get access to the download script. By filling out this form, you agree to the terms of use of the
  dataset.
- You can use the download script, which automatically download the entire dataset consisting of multiple .zip files, and unzippes them. Make sure you have "wget" and "unzip" installed. 
- Put the newly created MM-OR_data folder into the root directory of this project.
- Optionally download the 4D-OR dataset, download and put it to the root directory, and rename it 4D-OR_data. Instructions are in the official repo: https://github.com/egeozsoy/4D-OR. You can also find the newly annotated segmentations annotations and how to configure them in that repository.

## Panoptic Segmentation
This part of the repository contains the code for training and evaluating panoptic segmentation models. If you are only interested in the scene graph generation part, you can (mostly) skip this part.
For scene graph generation segmentations are also used, but this is an additional modality that can be skipped. Our paper provides results for both with and without segmentations.
This section builds upon the DVIS_PLUS repository (https://github.com/zhang-tao-whu/DVIS_Plus).

### Installation
- Make sure you run `git lfs install` and `git lfs pull` to download the large files.
- cd into panoptic_segmentation dir
- Run `pip install -r requirements.txt`. You might need to comment out the `detectron` line, and install it manually afterwards. Same for `panopticapi`.
- Install `detectron` by running `pip install "git+https://github.com/facebookresearch/detectron2.git@v0.6#egg=detectron2"`.
- Install `panopticapi` `pip install git+https://github.com/cocodataset/panopticapi.git`
- Follow the `mask2former` installation instructions, specifically regarding MSDeformAttn https://github.com/facebookresearch/Mask2Former/blob/main/INSTALL.md

### Training
- For single frame training on both dataset (4D-OR and MM-OR), simply run `python -u train_net_video.py --num-gpus 1 --config-file configs/dvis_Plus/HybridOR/CTVIS_r50.yaml --resume MODEL.WEIGHTS mask2former/ctvis_r50_vspw.pth SOLVER.IMS_PER_BATCH 1`
- After the single frame training, for online temporal training on both dataset (4D-OR and MM-OR), run `python -u train_net_video.py --num-gpus 1 --config-file configs/dvis_Plus/HybridOR/DVIS_Plus_Online_R50.yaml --resume MODEL.WEIGHTS output_CTVIS_R50_HybridOR_withsimstation/model_0099999.pth SOLVER.IMS_PER_BATCH 1`
- After the single frame training, for offline temporal training on both dataset (4D-OR and MM-OR), run `python -u train_net_video.py --num-gpus 1 --config-file configs/dvis_Plus/HybridOR/DVIS_Plus_Offline_R50.yaml --resume MODEL.WEIGHTS output_R50_HybridOR_temporal_online_withsimstation/model_0039999.pth SOLVER.IMS_PER_BATCH 1`

### Evaluation
- For evaluation any model after training, use the `--eval-only MODEL.WEIGHTS PATH_TO_MODEL.pth`, instead of `--resume MODEL.WEIGHTS PATH_TO_MODEL.pth`. Otherwise the command is the same as the training command.
- TODO put these into huggingface as well. We include in the repository all three checkpoints, which you can directly use. (
  e.g https://huggingface.co/egeozsoy/MM-OR/tree/main/output_CTVIS_R50_HybridOR_withsimstation, https://huggingface.co/egeozsoy/MM-OR/tree/main/output_R50_HybridOR_temporal_online_withsimstation, https://huggingface.co/egeozsoy/MM-OR/resolve/main/DVIS_Plus_Offline_R50_HybridOR_temporal_offline_52_reverseagu_withsimstation)
- Running a full evaluation like this will likely be necessary if you want to use the segmentations for scene graph generation. Make sure to run it for train/val and eventually test splits.


## Scene Graph Generation
This part of the repository contains the code for training and evaluating scene graph generation models. If you are only interested in the panoptic segmentation part, you can skip this part.
This section builds upon the ORacle (https://github.com/egeozsoy/ORacle) and LLava (https://github.com/haotian-liu/LLaVA) repositories.


### Installation
- Make sure you run `git lfs install` and `git lfs pull` to download the large files.
- cd into scene_graph_generation dir
- Run `pip install -r requirements.txt`.
- Locally install llava by going into LLaVa folder and running `pip install -e .`
- Potentially you need to explicitly install flash-attn like `pip install flash-attn --no-build-isolation`
- Install the correct spconv version for pointtransformers by running `pip install spconv-cu117`. Make sure to install the correct version for your CUDA version.
- Sometimes it might be necessary to force install the correct numpy version `pip install numpy==1.26.4`
- Install torch_scatter by following the direction at: https://github.com/rusty1s/pytorch_scatter. If using conda/miniconda it can be as simple as `conda install pytorch-scatter -c pyg`
- Some modalities need to be preprocessed before training. While our dataset already contains the preprocessed versions of these, if they are missing for whatever reason, you can use the following scripts to generate them `create_take_sample_audios.py`, `create_take_sample_audio_embeddings.py`, `create_take_sample_speech_transcripts.py`, `create_take_sample_segmasks.py` 

### Training
- For the training, we first need to generate the training json. To this end run `python -m scene_graph_prediction.llava_helpers.generate_dataset_format_for_llava`. Reading through this script is suggested, it has some parameters for adjusting number of samples via N_PERM, controling temporality and augmentations etc.
- Now with the training json ready, we can proceed to training. cd into the LLaVA folder and run:
```python
  python -m llava.train.train_mem \
  --lora_enable True \
  --bits 4 \
  --lora_r 128 \
  --lora_alpha 256 \
  --mm_projector_lr 2e-5 \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path liuhaotian/llava-v1.5-7b \
  --version v1 \
  --data_path ../data/llava_samples/train_20perm_Falsetemp_Truetempaug_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_drophistory0.5_MIXED.json \
  --token_weight_path ../data/llava_samples/train_token_freqs_7b_50perm.json \
  --image_folder / \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --learning_rate 2e-5 \
  --max_grad_norm 0.1 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --run_name llava-v1.5-7b-task-lora_hybridor_qlora_40perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED \
  --mv_type "learned" \
  --unfreeze_n_vision_tower_layers 12 \
  --do_img_order_augment \
  --do_multimodal_augment \
  --multimodal_drop_prop 0.50 \
  --do_augment False
```
- For temporal training instead, use the following command. Make sure to generate the corresponding json again using the first step, but you need to change the parameter ADD_TEMPORAL to True. Make sure to correctly point o the previous model in --curriculum_learning_weights:
```python
  python -m llava.train.train_mem \
  --lora_enable True \
  --bits 4 \
  --lora_r 128 \
  --lora_alpha 256 \
  --mm_projector_lr 2e-5 \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path liuhaotian/llava-v1.5-7b \
  --version v1 \
  --data_path ../data/llava_samples/train_20perm_Truetemp_Truetempaug_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_drophistory0.5_MIXED.json \
  --token_weight_path ../data/llava_samples/train_token_freqs_7b_50perm.json \
  --image_folder / \
  --vision_tower openai/clip-vit-large-patch14-336 \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED_temporal_curriculum \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --learning_rate 2e-5 \
  --max_grad_norm 0.1 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --run_name llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED_temporal_curriculum \
  --mv_type "learned" \
  --unfreeze_n_vision_tower_layers 12 \
  --do_img_order_augment \
  --do_multimodal_augment \
  --multimodal_drop_prop 0.50 \
  --do_augment False \
  --curriculum_learning_weights ./checkpoints/llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED \
  --mv_type "learned" \
  --unfreeze_n_vision_tower_layers 12 \
  --do_img_order_augment
```
- The pretrained scene graph generation models can be found at our hugingface repository: https://huggingface.co/egeozsoy/MM-OR,  under https://huggingface.co/egeozsoy/MM-OR/tree/main/checkpoints. Make sure to put them in their correct order. Pretrained scene graph generation models have to be unzipped and put under scene_graph_generation/LLaVa/checkpoints/.
- For downstream tasks, first generate data by running `python -m scene_graph_prediction.llava_helpers.generate_downstream_dataset_format_for_llava`, this will generate it for all 3 downstream tasks at once.
- Then train by running:
```python
  python -m llava.train.train_mem
  --lora_enable True \
  --bits 4 \
  --lora_r 128 \
  --lora_alpha 256 \
  --mm_projector_lr 2e-5 \
  --deepspeed ./scripts/zero2.json \
  --model_name_or_path liuhaotian/llava-v1.5-7b \
  --version v1 \
  --data_path ../data/llava_samples/downstream_task_train_Truetempaug_drophistory0.5.json \
  --image_folder / \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 True \
  --output_dir ./checkpoints/llava-v1.5-7b-task-lora_hybridor_qlora_downstream_tasks \
  --num_train_epochs 1 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --learning_rate 2e-5 \
  --max_grad_norm 0.1 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --lazy_preprocess True \
  --report_to wandb \
  --run_name llava-v1.5-7b-task-lora_hybridor_qlora_downstream_tasks \
  --mv_type "learned"
```


### Evaluation
- You can either train the models yourself our use our pretrained models.
- To evaluate a non temporal model run: `python -u -m scene_graph_prediction.main --config mmor.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED`. Make sure the model paths match.
- To evaluate a temporal model run: `python -u -m scene_graph_prediction.main --config mmor_temporal_pred.json --model_path LLaVA/checkpoints/llava-v1.5-7b-task-lora_hybridor_qlora_20perm_AZURE_SIMSTATION_TRACKERCAM_PC_AUDIO_SPEECH_ROBOTMETA_TRACKINGMETA_PREDSEGMASKS_0.50drop_MIXED_temporal_curriculum`. Make sure the model paths match.