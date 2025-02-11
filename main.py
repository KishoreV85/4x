import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN


def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=2)
    model.load_weights(2)
    for i, image in enumerate([f for f in os.listdir("inputs") if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]):
        image = Image.open(f"inputs/{image}").convert('RGB')
        sr_image = model.predict(image)
        output_path = os.path.join("results", f"{i}.png")
        print(f"Saved: {output_path}") 
        sr_image.save(f'results/{i}.png')


if __name__ == '__main__':
    main()
