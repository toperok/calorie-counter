class Config:
    SEED = 42

    TEXT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"


    TEXT_MODEL_UNFREEZE = "encoder.layer.11|pooler"
    IMAGE_MODEL_UNFREEZE = "blocks.6|conv_head|bn2"

    BATCH_SIZE = 32
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    CLASSIFIER_LR = 1e-3
    
    EPOCHS = 30
    DROPOUT = 0.15
    HIDDEN_DIM = 256
    
    OUTPUT_DIM = 1 

    NEDA_PATH = "data/dish.csv"
    CSV_PATH = "data/dish_eda.csv"
    INGR_CSV_PATH = "data/ingredients.csv"
    IMG_DIR = "data/images"
    SAVE_PATH = "best_model.pth"
    TRAIN_DF_PATH = "data/train_df.csv"
    VAL_DF_PATH = "data/val_df.csv"
    TEST_DF_PATH = "data/test_df.csv"
