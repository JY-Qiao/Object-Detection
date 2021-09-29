class Config:
    def __init__(self):
        self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]
        self.rpn_stride = 16

        self.num_rois = 128

        self.num_RPN_predict_pre = 300
        self.num_RPN_train_pre = 600

        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7
        
        self.classifier_min_overlap = 0
        self.classifier_max_overlap = 0.5
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        self.pooling_regions = 14