import requests

import gradio as gr
import torch
from monai.networks.nets import DenseNet121
import os
import numpy as np
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(2001)

LABELS = ['Normal', 'Pneumonia']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet121(spatial_dims=2, in_channels=3,
                    out_channels=2, pretrained=False)
model.load_state_dict(torch.load('42.pt', map_location=torch.device(device))['model_state_dict'])

model.eval()

transform = create_transform(
    **resolve_data_config({}, model=model)
)


def predict_fn(img):
    img = img.convert('RGB')
    # img = transform(img).unsqueeze(0)
    img.thumbnail((256,256), Image.ANTIALIAS)
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out = model(img)
    
    probabilites = torch.nn.functional.softmax(out[0], dim=0)

    values, indices = torch.topk(probabilites, k=2)

    return {LABELS[i]: v.item() for i, v in zip(indices, values)}

gr.Interface(predict_fn, gr.inputs.Image(type='pil'), outputs='label').launch()