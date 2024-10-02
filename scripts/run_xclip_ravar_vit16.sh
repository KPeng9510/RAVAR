# ViT-B/16
job_name="xclip_ravar_vit16"
 python -m torch.distributed.launch --nproc_per_node=4 \
    main_xclip.py --do_train --num_thread_reader=4 \
    --epochs=30 --batch_size=16 --n_display=50 \
    --output_dir ckpts3/${job_name} \
    --lr 1e-2 --max_words 32 --max_frames 12 --batch_size_val 64 \
    --datatype ravar \
    --feature_framerate 1 --coef_lr 1e-3 \
    --freeze_layer_num 0 --slice_framepos 2 --cross_model cross-base --loose_type \
    --loose_type --linear_patch 2d --sim_header seqTransf \
    --pretrained_clip_name ViT-B/16 2>&1 | tee -a log/${job_name}