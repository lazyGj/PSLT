_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection_3x.py',
    '_base_/schedules/schedule_1x.py',
    '_base_/default_runtime.py'
]
# optimizer
model = dict(
    pretrained='/home/gaojie/gaojie/gaojie/code/transformer/classification/ckpt/pslt.pth',
    backbone=dict(
        type='pslt',
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,  # freeze or not  # -1 0 1 2
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch'),
        neck=dict(
        # type='FPN',
        in_channels=[72, 144, 288, 576],
        # out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)