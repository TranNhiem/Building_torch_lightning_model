from config.absl_mock import Mock_Flag

#*********************************************
# Flags for Configure Training Experiment
#*********************************************

def base_cfg():
    
    flags = Mock_Flag()


    flags.DEFINE_integer(
        'image_size', 224,
        'image size.')

    flags.DEFINE_integer(
        'train_batch', 128,
        'Train batch_size .')

    flags.DEFINE_integer(
        'val_batch', 128,
        'Validaion_Batch_size.')

    flags.DEFINE_integer(
        'max_train_epochs', 100,
        'Number of epochs to train for.')
    
    flags.DEFINE_integer(
        'train_steps', 100,
        'Number of steps to train regargless to Val_ds, test_ds.')
    
    ## Directory for Configure Dataloader 

    flags.DEFINE_string(
        'train_path', '/data1/1K_New/train',
        'Train dataset path.')

    flags.DEFINE_string(
        'val_path', '/data1/1K_New/val',
        'Validaion dataset path.')
    
    flags.DEFINE_string(
        'test_path', '/data1/1K_New/val',
        'testing dataset path.')

  
  


    flags.DEFINE_string(
        'val_label', "ILSVRC2012_validation_ground_truth.txt",
        'val_label.')

#*********************************************
# Flags for Tracking - visualization results 
#*********************************************

def wandb_set():
    flags = Mock_Flag()
    
    flags.DEFINE_string(
        "wandb_project_name", "your project name?",
        "set the project name for wandb."
    )
    flags.DEFINE_string(
        "wandb_run_name", "Your experiment running name ?",
        "set the run name for wandb."
    )
    flags.DEFINE_enum(
        'wandb_mod', 'run', ['run', 'dryrun'],
        'update the to the wandb server or not')
    
    flags.DEFINE_string(
        'job_type', None, 
        'This Flags use for Grouping of experiment runing'
    )

#*********************************************
# Flags for Design - Choose Your Neural Net Architecture 
#*********************************************

def neural_net_architecture(): 
    flags = Mock_Flag()
    pass

#*********************************************
# Flags for Choosing your Hardware Resources
#*********************************************

def hardware_accelerate():
    flags = Mock_Flag()
    pass

def your_config():
    flags = Mock_Flag()
    base_cfg()
    wandb_set()
    neural_net_architecture()
    hardware_accelerate()
    
