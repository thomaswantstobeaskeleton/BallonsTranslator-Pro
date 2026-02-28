"""
MAT (Mask-Aware Transformer) inpainting – CVPR 2022, large-hole inpainting.
Requires MAT repo (github.com/fenglinglwb/MAT) and a pretrained checkpoint (Places/CelebA).
Uses generate_image.py with image + mask; 512×512. Optional: clone repo, download checkpoint, set repo_path.
"""
import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np
import cv2
from typing import List

from .base import InpainterBase, register_inpainter, TextBlock
from utils.imgproc_utils import resize_keepasp

MAT_SIZE = 512


@register_inpainter("mat")
class MATInpainter(InpainterBase):
    """
    MAT: Mask-Aware Transformer for large-hole inpainting (CVPR 2022).
    Requires MAT repo and checkpoint. Set repo_path and checkpoint_path. See https://github.com/fenglinglwb/MAT
    """
    inpaint_by_block = True
    check_need_inpaint = True

    params = {
        "repo_path": {
            "type": "line_editor",
            "value": "",
            "description": "Path to MAT repo (contains generate_image.py, networks/, dnnlib/).",
        },
        "checkpoint_path": {
            "type": "line_editor",
            "value": "",
            "description": "Path to MAT checkpoint .pth (e.g. places_512 or celebahq_256).",
        },
        "inpaint_size": {
            "type": "line_editor",
            "value": 512,
            "description": "Resize to this size for MAT (512 or 256).",
        },
        "description": "MAT inpainting (CVPR 2022). Clone MAT repo, download checkpoint, set repo_path.",
    }
    _load_model_keys = set()

    def __init__(self, **params) -> None:
        super().__init__(**params)

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_bin = (mask > 127).astype(np.uint8) * 255
        repo_path = (self.params.get("repo_path") or {}).get("value", "").strip()
        ckpt_path = (self.params.get("checkpoint_path") or {}).get("value", "").strip()
        if not repo_path or not os.path.isdir(repo_path):
            self.logger.warning("MAT: repo_path not set or invalid. Set path to MAT repo (github.com/fenglinglwb/MAT).")
            return img.copy()
        if not ckpt_path or not os.path.isfile(ckpt_path):
            self.logger.warning("MAT: checkpoint_path not set or file not found. Download from MAT README.")
            return img.copy()
        gen_py = os.path.join(repo_path, "generate_image.py")
        if not os.path.isfile(gen_py):
            self.logger.warning("MAT: generate_image.py not found in repo_path.")
            return img.copy()

        size = int((self.params.get("inpaint_size") or {}).get("value", 512))
        im_h, im_w = img.shape[:2]
        if max(im_h, im_w) > size:
            img_s = resize_keepasp(img, size, stride=None)
            mask_s = resize_keepasp(mask_bin, size, stride=None)
        else:
            img_s = img
            mask_s = mask_bin
        h, w = img_s.shape[:2]
        img_512 = cv2.resize(img_s, (MAT_SIZE, MAT_SIZE), interpolation=cv2.INTER_LANCZOS4)
        mask_512 = cv2.resize(mask_s, (MAT_SIZE, MAT_SIZE), interpolation=cv2.INTER_NEAREST)

        tmp = tempfile.mkdtemp(prefix="mat_inpaint_")
        try:
            input_dir = os.path.join(tmp, "input")
            mask_dir = os.path.join(tmp, "mask")
            out_dir = os.path.join(tmp, "out")
            for d in (input_dir, mask_dir, out_dir):
                os.makedirs(d, exist_ok=True)
            base_name = "img.png"
            cv2.imwrite(os.path.join(input_dir, base_name), cv2.cvtColor(img_512, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(mask_dir, base_name), mask_512)

            # MAT generate_image.py CLI may vary; try common patterns
            cmd = [
                sys.executable,
                gen_py,
                "--ckpt", ckpt_path,
                "--input", input_dir,
                "--mask", mask_dir,
                "--output", out_dir,
            ]
            env = os.environ.copy()
            env["PYTHONPATH"] = repo_path if not env.get("PYTHONPATH") else repo_path + os.pathsep + env["PYTHONPATH"]
            result = subprocess.run(
                cmd,
                cwd=repo_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                # Try alternative args (some repos use --path for ckpt, --image_dir, etc.)
                cmd = [
                    sys.executable, gen_py,
                    "--path", ckpt_path,
                    "--input", input_dir,
                    "--mask", mask_dir,
                    "--output", out_dir,
                ]
                result = subprocess.run(cmd, cwd=repo_path, env=env, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                self.logger.error(f"MAT generate_image.py failed: {result.stderr or result.stdout}")
                return img.copy()

            out_path = os.path.join(out_dir, base_name)
            if not os.path.isfile(out_path):
                for f in os.listdir(out_dir):
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        out_path = os.path.join(out_dir, f)
                        break
            if not os.path.isfile(out_path):
                self.logger.error("MAT: no output image produced.")
                return img.copy()
            out_img = cv2.imread(out_path)
            if out_img is None:
                out_img = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
            if out_img is None:
                return img.copy()
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            out_img = cv2.resize(out_img, (w, h), interpolation=cv2.INTER_LANCZOS4)
            if (h, w) != (im_h, im_w):
                out_img = cv2.resize(out_img, (im_w, im_h), interpolation=cv2.INTER_LANCZOS4)
            mask_orig = (mask > 127).astype(np.float32)[:, :, np.newaxis]
            result_final = (out_img.astype(np.float32) * mask_orig + img.astype(np.float32) * (1 - mask_orig)).astype(np.uint8)
            return result_final
        except subprocess.TimeoutExpired:
            self.logger.error("MAT: generate_image.py timed out.")
        except Exception as e:
            self.logger.error(f"MAT inpainting failed: {e}")
        finally:
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass
        return img.copy()
