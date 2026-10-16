#!/usr/bin/env python3
"""Create independent preprocessing fixtures from pinned HF resize math and Pillow."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct
import math
from PIL import Image, ImageDraw, ImageFont
import PIL

root=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument("--output",type=Path,default=root/"tests/vision/fixtures")
a=p.parse_args()
lock=json.loads((root/"tests/vision/manifest.json").read_text())['references']
if PIL.__version__ != lock['pillow_version']:
    p.error(f"requires Pillow {lock['pillow_version']}; found {PIL.__version__}")
source=root/".vision-cache/reference/image_processing_gemma4.py"
if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != lock['source_sha256'][source.name]:
    p.error("pinned resize source missing or mismatched; run tools/vision_artifacts.py --references --download")
tree=ast.parse(source.read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="get_aspect_ratio_preserving_size")
scope={"math":math};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),"exec"),scope)
resize=scope[fn.name]
a.output.mkdir(parents=True,exist_ok=True)
manifest={"pillow_version":PIL.__version__,"resize_source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"cases":[]}
for name,w,h in [("quadrants",1920,1344),("wide",1031,173),("portrait",173,1031)]:
    im=Image.new("RGB",(w,h));pixels=im.load()
    for y in range(h):
        for x in range(w):
            # Deterministic sharp edges and gradients exercise the resampler.
            colors=[(255,0,0),(0,255,0),(0,0,255),(255,255,0)]
            base=colors[(x>=w//2)+2*(y>=h//2)]
            pixels[x,y]=base if name=="quadrants" else ((x*13+y*7)%256,(x*3+y*17)%256,(x*19+y)%256)
    path=a.output/(name+".png");im.save(path)
    for budget in (70,140,280,560,1120):
        th,tw=resize(h,w,16,budget*9,3)
        rgb=im.resize((tw,th),Image.Resampling.BICUBIC).tobytes()
        prepared=a.output/f"{name}-{budget}.rgb"
        prepared.write_bytes(struct.pack("<II",tw,th)+rgb)
        manifest["cases"].append({"image":path.name,"budget":budget,"width":tw,"height":th,
                                  "rows":tw//48*(th//48),"prepared":prepared.name,
                                  "sha256":hashlib.sha256(prepared.read_bytes()).hexdigest()})
# A readable text fixture exercises the screenshot/OCR path independently of
# the colour prompt. Pixels are retained with their hash, not a font path.
im=Image.new("RGB",(640,256),"white")
ImageDraw.Draw(im).text((85,90),"CODE: 7318",font=ImageFont.load_default(size=64),fill="black")
path=a.output/"code.png";im.save(path)
manifest["text_fixture"]={"image":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"expected":"7318"}
path=a.output/"quadrants-rotated.png"
Image.open(a.output/"quadrants.png").transpose(Image.Transpose.ROTATE_180).save(path)
manifest["rotated_fixture"]={"image":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"expected_upper_left":"yellow"}
(a.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
