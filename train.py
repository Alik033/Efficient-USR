import argparse
import os
import shutil

from torch.backends import cudnn
from utilities.json_config import readConfig, writeConfig
from utilities.reporter import Reporter
from utilities.yaml_config import getConfigYaml

####################################################################################
# To configure the seting of training\finetune\test
####################################################################################
def getParameters():
    
    parser = argparse.ArgumentParser()
    # general settings
    parser.add_argument('-v', '--version', type=str, default='UIE_UFO120_CA',
                                            help="version name for train, finetune")

    parser.add_argument('-p', '--phase', type=str, default="train",
                                            choices=['train', 'finetune'],
                                                help="The phase of current project")

    parser.add_argument('-c', '--cuda', type=int, default=0) # <0 if it is set as -1, program will use CPU
    parser.add_argument('-e', '--ckpt', type=int, default=74,
                                help="checkpoint epoch for test phase or finetune phase")

    # training
    parser.add_argument('--experiment_description', type=str,
                                default="traing SR")

    parser.add_argument('--train_yaml', type=str, default="train_USR248.yaml")

    return parser.parse_args()


####################################################################################
# This function will create the related directories before the 
# training\fintune\test starts
# Your_log_root (version name)
#   |---checkpoints/...
#   |---scripts/...
####################################################################################
def createDirs(sys_state):
    # the base dir
    if not os.path.exists(sys_state["log_root_path"]):
        os.makedirs(sys_state["log_root_path"])

    # create dirs
    sys_state["project_root"]        = os.path.join(sys_state["log_root_path"],
                                            sys_state["version"])
                                            
    project_root                     = sys_state["project_root"]
    if not os.path.exists(project_root):
        os.makedirs(project_root)
    
    sys_state["project_checkpoints"] = os.path.join(project_root, "checkpoints")
    if not os.path.exists(sys_state["project_checkpoints"]):
        os.makedirs(sys_state["project_checkpoints"])

    sys_state["project_scripts"]     = os.path.join(project_root, "scripts")
    if not os.path.exists(sys_state["project_scripts"]):
        os.makedirs(sys_state["project_scripts"])
    
    sys_state["reporter_path"] = os.path.join(project_root,sys_state["version"]+"_report")

def main():
    config = getParameters()
    # speed up the program
    cudnn.benchmark = True

    sys_state = {}

    # set the GPU number
    if config.cuda >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda)

    # read system environment paths
    env_config = readConfig('env/env.json')
    env_config = env_config["path"]

    # obtain all configurations in argparse
    ignore_key = []
    config_dic = vars(config)
    for config_key in config_dic.keys():
        sys_state[config_key] = config_dic[config_key]
        ignore_key.append(config_key)

    #=======================Train Phase=========================#
    if config.phase == "train":
        # read training configurations from yaml file
        ymal_config = getConfigYaml(os.path.join(env_config["train_config_path"], config.train_yaml))
        for item in ymal_config.items():
            sys_state[item[0]] = item[1]

        # create related dirs
        sys_state["log_root_path"] = env_config["train_log_root"]
        createDirs(sys_state)

        # save the config json
        config_json = os.path.join(sys_state["project_root"], env_config["config_json_name"])
        writeConfig(config_json, sys_state)
        
        # save the dependent scripts 
        # TODO and copy the scripts to the project dir
        
        # save the trainer script into [train_logs_root]\[version name]\scripts\
        file1       = os.path.join(env_config["train_scripts_path"],
                                    "trainer_%s.py"%sys_state["train_script_name"])
        tgtfile1    = os.path.join(sys_state["project_scripts"],
                                    "trainer_%s.py"%sys_state["train_script_name"])
        shutil.copyfile(file1,tgtfile1)


        # save the yaml file
        file1       = os.path.join(env_config["train_config_path"], config.train_yaml)
        tgtfile1    = os.path.join(sys_state["project_scripts"], config.train_yaml)
        shutil.copyfile(file1,tgtfile1)

    # get the dataset path
    sys_state["dataset_paths"] = {}
    for data_key in env_config["dataset_paths"].keys():
        sys_state["dataset_paths"][data_key] = env_config["dataset_paths"][data_key]
    
    sys_state["test_dataset_paths"] = {}
    for data_key in env_config["test_dataset_paths"].keys():
        sys_state["test_dataset_paths"][data_key] = env_config["test_dataset_paths"][data_key]

    # display the training information
    if config.phase == "train":
        moduleName  = "train_scripts.trainer_" + sys_state["train_script_name"]
    if config.phase == "finetune":
        finetune_configs = readConfig(os.path.join(env_config["train_log_root"], config.version, env_config["config_json_name"]))
        for item in finetune_configs.items():
            if item[0] in ignore_key:
                pass
            else:
                sys_state[item[0]] = item[1]
        moduleName  = "train_logs." + config.version + ".scripts.trainer_" + sys_state["train_script_name"]

    # print some important information
    # TODO
    print("Start to run training script: {}".format(moduleName))
    print("Traning version: %s"%sys_state["version"])
    print("Dataloader Name: %s"%sys_state["dataloader"])
    # print("Image Size: %d"%sys_state["imsize"])
    print("Batch size: %d"%(sys_state["batch_size"]))

    # create reporter file
    reporter = Reporter(sys_state["reporter_path"])
    # Load the training script and start to train
    reporter.writeConfig(sys_state) 
    

    package     = __import__(moduleName, fromlist=True)
    trainerClass= getattr(package, 'Trainer')
    trainer     = trainerClass(sys_state, reporter)
    trainer.train()



if __name__ == '__main__':
    main()