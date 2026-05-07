from __future__ import annotations

import argparse
from pathlib import Path

from .detector import AIFaceDetector
from .manual_roi import parse_box
from .visualization import save_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect whether a face image is AI-generated.")
    parser.add_argument("image", help="Input image path.")
    parser.add_argument("--model", default=None, help="Optional trained model path.")
    parser.add_argument("--save-dir", default=None, help="Save annotated and evidence images.")
    parser.add_argument("--box", default=None, help="Manual face box as x,y,w,h in original image pixels.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args()

    detector = AIFaceDetector(args.model)
    manual_box = parse_box(args.box) if args.box else None
    result = detector.analyze(args.image, manual_box=manual_box)

    if args.json:
        print(result.to_json())
    else:
        print(f"结论: {result.label}")
        print(f"AI 生成概率: {result.ai_probability * 100:.2f}%")
        print(f"真实拍摄概率: {result.real_probability * 100:.2f}%")
        print(f"模式: {result.model_mode}")
        print(f"人脸区域: {result.face_note}")
        for item in result.evidence:
            print(f"- {item}")

    if args.save_dir:
        output = Path(args.save_dir)
        stem = Path(args.image).stem
        save_rgb(output / f"{stem}_annotated.jpg", result.annotated_image)
        save_rgb(output / f"{stem}_residual.jpg", result.residual_image)
        save_rgb(output / f"{stem}_frequency.jpg", result.frequency_image)
        (output / f"{stem}_report.json").write_text(result.to_json(), encoding="utf-8")
        print(f"已保存输出到: {output}")


if __name__ == "__main__":
    main()
