#!/usr/bin/env python3
"""Start the validated Italian-steering pilot in the native interactive chat."""
import argparse
import json
import math
import os
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    selected=json.loads((root/'steering/examples/language/selected.json').read_text())[0]
    p=argparse.ArgumentParser(description=__doc__,epilog='Default model path matches xenolith-ptl; override --model elsewhere.')
    p.add_argument('--model',type=Path,default=Path.home()/'models/gemma-4-26B-A4B-it-qat/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf')
    p.add_argument('--manifest',type=Path,default=root/'steering/local/italian/manifest.json')
    p.add_argument('--baseline',action='store_true',help='Load the same artifact at zero strength')
    p.add_argument('--strength',type=float,help='Override the pilot strength; requires separate evaluation')
    p.add_argument('-n',type=int,default=512,help='Maximum output tokens per response')
    a=p.parse_args()
    strength=0.0 if a.baseline else selected['strength'] if a.strength is None else a.strength
    if not math.isfinite(strength) or a.n<1:p.error('Expected finite strength and positive output budget')
    if not a.model.is_file():p.error('Model missing; pass --model /path/to/the/exact.gguf')
    if not a.manifest.is_file():p.error('Pilot artifact missing; see steering/LANGUAGE_EVALUATION.md')
    binary=root/'xenolith'
    if not os.access(binary,os.X_OK):p.error('Build the native executable with make on Linux/Intel')
    print(f'Italian steering pilot, strength {strength:.9g}. Ask a question in English.\n'
          f'Use /steer off, /steer {selected["strength"]:.9g}, or /exit.\n'
          'Turning steering off retains historical cache/text effects; restart with --baseline for a fresh comparison.',flush=True)
    os.execv(str(binary),[str(binary),'chat',str(a.model),'--ctx','4096','--steering',str(a.manifest),
        '--steering-strength',str(strength),'--temp','0','--seed','1','-n',str(a.n)])


if __name__=='__main__':main()
