import os
import urllib.request
import numpy as np
import dezero
import dezero.models
import dezero.functions as F
from dezero import Variable

def fetch_vgg16_weights():
    vgg = dezero.models.VGG16(pretrained=True)
    
    params = {}
    vgg._flatten_params(params)
    # vgg._flatten_params maps to Parameter objects. We need their .data.
    data_params = {k: v.data for k, v in params.items()}
    
    os.makedirs('dataset/weights', exist_ok=True)
    np.savez('dataset/weights/vgg16.npz', **data_params)
    print("Saved dataset/weights/vgg16.npz with weights.")
    
    np.random.seed(42)
    x_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
    x = Variable(x_data)
    
    with dezero.test_mode():
        y = vgg(x)
        
    np.savez('dataset/weights/vgg16_test.npz', x=x_data, y=y.data)
    print("Saved dataset/weights/vgg16_test.npz with input 'x' and expected output 'y'.")

if __name__ == '__main__':
    fetch_vgg16_weights()
