

import os
import torch

class Config:
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, r"C:\Users\user\Desktop\genai project\project_5\assessment_exercise_data\assessment-data ")
    
   
    BATCH_SIZE = 32
    NUM_EPOCHS = 25
    LEARNING_RATE = 0.001
    NUM_CLASSES = 100  
    PRETRAINED = True  

    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    VAL_SPLIT = 0.2
    MODEL_SAVE_PATH = "best_model.pth"
