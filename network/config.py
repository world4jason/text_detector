import datetime
import numpy as np
class Config(object):
    # Dataset Setting
    DATASET_NAME = "SynthText"
    VALIDATIO_DATASET_PATH = 'val_quad_norm.record'
    TRAIN_DATASET_PATH = 'train_quad_norm.record'

    # Harware Setting
    NUM_GPU = 1
    IMAGES_PER_GPU = 32

    # Training Setting
    NUM_TRAINING_STEPS = 10000000
    LEARNING_RATE = 0.001
    LEARNING_DECAY_STEPS = [40000, 60000]
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001
    OPTIMIZER = "Adam"

    USE_VALIDATION = True
    VALIDATION_BATCH_SIZE = 1
    VALIDATION_STEPS = 25

    MOVING_AVERAGE_DECAY = 0.9999
    CLIP_NORM = 5

    TRAINABLE_SCOPE = "resnet_backbone|fpn|task_head"

    # Image Setting
    IMAGE_SHAPE = (640,640)

    # Network Setting
    BACKBONE = "resnet50"
    NUM_CLASSES = 1+1


    FEATURE_STRIDES =  [8, 16, 32, 64, 128]

    DETECTION_MIN_CONFIDENCE = 0.7
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MAX_OBJECTS = 300

    USE_BN = True
    TRAIN_BN = False


    # Augment Setting
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    VAR_PIXEL = np.array([58.395, 57.12, 57.375])

    # ANCHOR Setting
    ASPECT_RATIOS = [1., 2., 3., 5., 1./2., 1./3., 1./5.]
    ANCHOR_AREAS = [32, 64, 128, 256, 512] #[16,32,64,128,256]

    # ANCHOR STRIDE
    ANCHOR_VERTICAL_STEP = 0.5
    ANCHOR_HORIZON_STEP = 1

    OBJECTS_THRESHOLD = 0.5
    BACKGROUND_THRESHOLD = 0.4


    # Summary Setting
    SUMMARY_STEPS = 100

    # Decode
    CLS_THRESH = 0.6
    NMS_THRESH = 0.95
    PNMS_THRESH = 0.99
    NUM_OUTPUT_BOXES = 300


    # Logs
    OUTPUT_LOGS = "output_logs/"+"{}{:%Y%m%dT%H%M}".format( DATASET_NAME+"_", datetime.datetime.now())+"/"

    #Loss
    LOSS_WEIGHTS = {"class_loss":1,"loc_loss":0.2}

    #Train
    NUM_THREADS = 8
    MIN_AFTER_DEQUEUE = 40

    # Evaluate
    COCO_mAP_RANGE = np.round(np.arange(0.5,1.,0.05,np.float32),4)

    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.NUM_GPU*self.IMAGES_PER_GPU

        # Effective anchor grid
        self.NUM_ANCHOR_PER_GRID = len(self.ASPECT_RATIOS)*(1/self.ANCHOR_VERTICAL_STEP)*(1/self.ANCHOR_HORIZON_STEP)

        #min_after_dequeue + (num_threads + a small safety margin) * batch_size

        self.CAPACITY=self.MIN_AFTER_DEQUEUE+(self.NUM_THREADS+self.NUM_THREADS)*self.BATCH_SIZE



    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


config = Config()
