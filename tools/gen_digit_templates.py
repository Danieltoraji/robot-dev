# -*- coding: utf-8 -*-
"""用字体程序化生成数字模板（白字黑底），供 vision.DigitRecognizer 使用。

用法：
  python tools/gen_digit_templates.py --out templates/arial
  python tools/gen_digit_templates.py --out templates/arial --font C:/Windows/Fonts/arialbd.ttf --size 40

生成 templates/<out>/0.png ~ 9.png。建议生成多套（不同字体/字号），
再用 debug_vision.py digit 逐套试，选匹配分数最高的一套。

依赖：Pillow（仅此开发工具需要，机器人运行不依赖）。
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="生成数字识别模板（白字黑底）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--font", default=None, help="字体文件路径；缺省用 PIL 默认字体")
    ap.add_argument("--size", type=int, default=40, help="渲染字号(像素)")
    ap.add_argument("--pad", type=int, default=6, help="内边距(像素)")
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(args.font, args.size) if args.font else ImageFont.load_default()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ch in "0123456789":
        # 先量出文字实际尺寸，再做紧凑画布
        tmp = Image.new("L", (args.size * 2, args.size * 2), 0)
        ImageDraw.Draw(tmp).text((0, 0), ch, fill=255, font=font)
        bbox = tmp.getbbox()
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        canvas = Image.new("L", (w + args.pad * 2, h + args.pad * 2), 0)
        ImageDraw.Draw(canvas).text((args.pad - bbox[0], args.pad - bbox[1]), ch, fill=255, font=font)
        canvas.save(out_dir / f"{ch}.png")
        print(f"已生成 {out_dir / (ch + '.png')}")


if __name__ == "__main__":
    main()
