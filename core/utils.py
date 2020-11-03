import cv2
import torch
import numpy as np

def get_one_hot_vector(label, classes):
    vector = np.zeros((classes), dtype = np.float32)
    if len(label) > 0:
        vector[label] = 1.
    return vector

def get_learning_rate_from_optimizer(optimizer):
    return optimizer.param_groups[0]['lr']

def get_numpy_from_tensor(tensor):
    return tensor.cpu().detach().numpy()

def load_model(model, model_path):
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(model_path))
    else:
        model.load_state_dict(torch.load(model_path))

def save_model(model, model_path):
    if torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)
