# IMPORTANT: RUN AS SOON AS A NEW .png FILE IS CREATED!!!!!!

from pathlib import Path
from PIL import Image

ROOT = Path("/Users/marcel/PycharmProjects/Master_Thesis_Pavia")

for p in ROOT.rglob("*.png"):
    im = Image.open(p)
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGBA", im.size, "white")
        Image.alpha_composite(bg, im.convert("RGBA")).convert("RGB").save(p)
        print("fixed:", p.name)