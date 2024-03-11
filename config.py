from easydict import EasyDict as edict

config = edict()
config.dataset = "webface" # training dataset
config.embedding_size = 512 # embedding size of evaluation
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128 # batch size per GPU
config.lr = 0.1
config.output = "output/R50_CRFIQA" # train evaluation output folder
config.global_step=0 # step to resume
config.s=64.0
config.m=0.50
config.beta=0.5



# type of network to train [ iresnet100 | iresnet50 ]
config.network = "iresnet50"




if config.dataset == "emoreIresNet":
    config.rec = "datafaces_emore"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch =  18
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw"]
    config.eval_step=5686
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8, 14,20,25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "data/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 34   #  [22, 30, 35] [22, 30, 40]
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30"]
    config.eval_step= 958 #33350

    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [20, 28, 32] if m - 1 <= epoch])
    config.lr_func = lr_step_func
