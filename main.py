from diffusers import DiffusionPipeline
from PIL import Image
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

images = pipe(prompt=prompt).images[0]

# 将 Tensor 转换为 PIL 图像对象
image_pil = Image.fromarray(images.permute(1, 2, 0).cpu().numpy().astype('uint8'))

# 保存图像
image_pil.save('astronaut_green_horse.png')

print("图像保存成功！")
"""
FutureWarning: `Transformer2DModelOutput` is deprecated and will be removed in version 1.0.0. 
Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version.
Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead.
deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)
"""