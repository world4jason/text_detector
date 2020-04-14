import datetime
import numpy as np
class Config(object):
    ###################
    # Dataset Setting #
    ###################
    # DataSet Name
    DATASET_NAME = "SynthText"
    # Path to tensorflow record
    VALIDATIO_DATASET_PATH = 'val_quad_norm.record'
    TRAIN_DATASET_PATH = 'train_quad_norm.record'

    ###################
    # Harware Setting #
    ###################
    # NUMBER OF GPUs to use.
    NUM_GPU = 1
    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    IMAGES_PER_GPU = 8

    ####################
    # Training Setting #
    ####################
    # Augmentation
    AUGMENT = True
    # Total Training Step
    NUM_TRAINING_STEPS = 10000000
    # Learning rate, Learning rate strategy and momentum
    LEARNING_RATE = 0.001
    LEARNING_DECAY_STEPS = [40000, 60000]
    LEARNING_MOMENTUM = 0.9
    OPTIMIZER = "Momentum"
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Batch utility setting
    NUM_THREADS = 8
    MIN_AFTER_DEQUEUE = 40

    ######################
    # Validation Setting #
    ######################
    USE_VALIDATION = True
    VALIDATION_BATCH_SIZE = 1
    VALIDATION_STEPS = 25

    MOVING_AVERAGE_DECAY = 0.9999
    # Gradient norm clipping
    CLIP_NORM = 5

    TRAINABLE_SCOPE = "resnet_backbone|fpn|task_head"


    ###################
    # Network Setting #
    ###################
    # Image Setting
    IMAGE_SHAPE = (640,640)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    VAR_PIXEL = np.array([58.395, 57.12, 57.375])
    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"
    NUM_CLASSES = 1+1
    C2_MODE = False

    # USE BatchNorm or GroupNorm
    USE_BN = True
    # Train BatchNorm
    TRAIN_BN = False

    # The stride of each layer output from fpn
    # Based on a Resnet101 backbone, where BACKBONE_STRIDES = {"C1":2,"C2":4,"C3":8,"C4":16,"C5":32}
    FEATURE_STRIDES =  [8, 16, 32, 64, 128]

    ##################
    # ANCHOR Setting #
    ##################
    # Bounding box refinement standard deviation.
    # Same as proir variance in Single Shot Network(SSD)
    PRIOR_VARIANCE = False
    RECT_BBOX_XY_PRIOR_VARIANCE = np.array([0.1,0.1])
    RECT_BBOX_WH_PRIOR_VARIANCE = np.array([0.2,0.2])
    QUAD_BBOX_PRIOR_VARIANCE = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    ANCHOR_RATIOS = [1., 2., 3., 5., 1./2., 1./3., 1./5.]
    # Length of square anchor side in pixels
    ANCHOR_AREAS = [32, 64, 128, 256, 512] #[16,32,64,128,256]

    # ANCHOR STRIDE
    ANCHOR_VERTICAL_STEP = 0.5
    ANCHOR_HORIZON_STEP = 1

    ##################
    # Encode Setting #
    ##################
    # OBJECTS_THRESHOLD < Objects
    # BACKGROUND_THRESHOLD > Backgrounds
    # OBJECTS_THRESHOLD > ignore > OBJECTS_THRESHOLD
    OBJECTS_THRESHOLD = 0.5
    BACKGROUND_THRESHOLD = 0.4



    # Summary Setting
    SUMMARY_STEPS = 100

    ##################
    # Decode Setting #
    ##################
    CLS_THRESH = 0.6
    NMS_THRESH = 0.95
    PNMS_THRESH = 0.99
    NUM_OUTPUT_BOXES = 300


    # Logs
    OUTPUT_LOGS = "output_logs/"+"{}{:%Y%m%dT%H%M}".format( DATASET_NAME+"_", datetime.datetime.now())+"/"

    # Loss weights for more precise optimization.
    LOSS_WEIGHTS = {"class_loss":1,"loc_loss":0.2}



    # Evaluate
    COCO_mAP_RANGE = np.round(np.arange(0.5,1.,0.05,np.float32),4)

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.NUM_GPU*self.IMAGES_PER_GPU

        # Effective anchor grid
        self.NUM_ANCHOR_PER_GRID = len(self.ANCHOR_RATIOS)*(1/self.ANCHOR_VERTICAL_STEP)*(1/self.ANCHOR_HORIZON_STEP)

        #min_after_dequeue + (num_threads + a small safety margin) * batch_size

        self.CAPACITY=self.MIN_AFTER_DEQUEUE+(self.NUM_THREADS+self.NUM_THREADS)*self.BATCH_SIZE

        if self.C2_MODE:
            self.FEATURE_STRIDES =  [4, 8, 16, 32, 64, 128]
            self.ANCHOR_AREAS = [16, 32, 64, 128, 256, 512]



    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = Config()
