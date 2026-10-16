#!/usr/bin/env python3
"""Compare native PNG/EXIF/alpha pixels with pinned Pillow and HF resize math."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess

from check_support import check, fresh_output, write_report


root = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--prepare', type=Path, default=root / 'tools/vision_prepare')
p.add_argument('--reference', type=Path,
               default=root / '.vision-cache/reference/image_processing_gemma4.py',
               help='Pinned HF image processing source from vision_artifacts.py --references')
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output = fresh_output(a.output)
report = {'status': 'RUNNING', 'cases': []}
write_report(a.output, report)
try:
    import PIL
    from PIL import Image, ImageOps

    locked = json.loads((root / 'tests/vision/manifest.json').read_text())['references']
    check(PIL.__version__ == locked['pillow_version'],
          ('Pillow version mismatch', PIL.__version__, locked['pillow_version']))
    source = a.reference.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    check(source_hash == locked['source_sha256']['image_processing_gemma4.py'],
          ('HF reference checksum mismatch', a.reference, source_hash))
    report.update(pillow_version=PIL.__version__, reference_sha256=source_hash)
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'get_aspect_ratio_preserving_size')
    scope = {'math': math}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(a.reference), 'exec'), scope)
    resize = scope[fn.name]
    im = Image.new('RGB', (43, 29))
    im.putdata([((x*17+y*5)%256, (x*3+y*23)%256, (x*13+y*7)%256)
                for y in range(29) for x in range(43)])
    cases = []
    for orientation in range(1, 9):
        exif = Image.Exif()
        exif[274] = orientation
        path = a.output / f'exif-{orientation}.png'
        im.save(path, exif=exif)
        cases.append((path, 70, 0))
    exif = Image.Exif()
    exif[274] = 6
    path = a.output / 'oriented.jpg'
    im.save(path, exif=exif, quality=95, subsampling=0)
    cases.append((path, 70, 3))
    alpha = Image.new('RGBA', (43, 29))
    alpha.putdata([(x*5%256, y*7%256, (x+y)*9%256, (x*7+y*11)%256)
                   for y in range(29) for x in range(43)])
    path = a.output / 'alpha.png'
    alpha.save(path)
    cases.append((path, 70, 0))
    for name in ('quadrants', 'wide', 'portrait'):
        for budget in (70, 140, 280, 560, 1120):
            cases.append((root / 'tests/vision/fixtures' / (name + '.png'), budget, 0))
    for path, budget, tolerance in cases:
        with Image.open(path) as opened:
            reference = ImageOps.exif_transpose(opened).convert('RGBA')
        background = Image.new('RGBA', reference.size, 'white')
        reference = Image.alpha_composite(background, reference).convert('RGB')
        w, h = reference.size
        th, tw = resize(h, w, 16, budget*9, 3)
        expected = reference.resize((tw, th), Image.Resampling.BICUBIC).tobytes()
        output = a.output / f'{path.stem}-{budget}.rgb'
        subprocess.run([str(a.prepare.resolve()), str(path), str(budget), str(output)],
                       check=True, stdout=subprocess.DEVNULL)
        actual = output.read_bytes()
        check(struct.unpack('<II', actual[:8]) == (tw, th), (path, budget, 'dimensions'))
        check(len(actual[8:]) == len(expected), (path, budget, 'pixel count'))
        maximum = max(abs(x-y) for x, y in zip(actual[8:], expected))
        report['cases'].append({'image': path.name, 'budget': budget,
                                'max_pixel_difference': maximum, 'tolerance': tolerance,
                                'pass': maximum <= tolerance,
                                'input_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(path.name, budget, maximum, 'PASS' if maximum <= tolerance else 'FAIL')
    # Truncation must fail without returning an image.
    bad = a.output / 'truncated.png'
    bad.write_bytes((root / 'tests/vision/fixtures/quadrants.png').read_bytes()[:64])
    rejected = subprocess.run([str(a.prepare.resolve()), str(bad), '70',
                               str(a.output / 'bad.rgb')], capture_output=True)
    check(rejected.returncode != 0, 'truncated input was accepted')
    report['truncated_input_rejected'] = True
    check(all(row['pass'] for row in report['cases']), 'pixel comparison failed')
    report['status'] = 'PASS'
except BaseException as exc:
    report['status'] = 'FAIL'
    report['error'] = repr(exc)
    raise
finally:
    write_report(a.output, report)
