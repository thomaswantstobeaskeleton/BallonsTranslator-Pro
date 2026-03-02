import numpy as np
import cv2
from typing import Dict, List
from collections import OrderedDict
import sys

from utils.registry import Registry
from utils.textblock_mask import extract_ballon_mask
from utils.imgproc_utils import enlarge_window
from utils.config import pcfg

from ..base import BaseModule, DEFAULT_DEVICE, soft_empty_cache, DEVICE_SELECTOR, GPUINTENSIVE_SET, TORCH_DTYPE_MAP, BF16_SUPPORTED
from ..textdetector import TextBlock

INPAINTERS = Registry('inpainters')
register_inpainter = INPAINTERS.register_module


def _feather_weight_2d(h: int, w: int, feather_px: int) -> np.ndarray:
    """Weight mask for blending: 1 in center, smooth falloff to 0 at edges. Reduces visible paste seams."""
    if feather_px <= 0 or h < 3 or w < 3:
        return np.ones((h, w), dtype=np.float32)
    fy = np.linspace(0, 1, h, dtype=np.float32)
    fx = np.linspace(0, 1, w, dtype=np.float32)
    # Raised cosine: 1 in center, 0 at edges over feather_px
    def fade(edge0: int, edge1: int, n: int) -> np.ndarray:
        out = np.ones(n, dtype=np.float32)
        if edge0 > 0:
            t = np.linspace(1, 0, edge0, dtype=np.float32)
            out[:edge0] = 0.5 * (1 + np.cos(np.pi * t))
        if edge1 < n:
            t = np.linspace(0, 1, n - edge1, dtype=np.float32)
            out[edge1:] = 0.5 * (1 + np.cos(np.pi * t))
        return out
    wy = fade(feather_px, h - feather_px, h)
    wx = fade(feather_px, w - feather_px, w)
    w2d = wy[:, np.newaxis] * wx[np.newaxis, :]
    return w2d


def inpaint_handle_alpha_channel(original_alpha, mask):
    '''
    perhaps a better idea is to feed the alpha into inpainting model, but it'll double the cost  
    for now it just return the original alpha
    '''

    result_alpha = original_alpha.copy()

    # Analyze the alpha values around the original mask to determine appropriate transparency
    mask_dilated = cv2.dilate((mask > 127).astype(np.uint8), np.ones((15, 15), np.uint8), iterations=1)
    surrounding_mask = mask_dilated - (mask > 127).astype(np.uint8)

    if np.any(surrounding_mask > 0):
        surrounding_alpha = original_alpha[surrounding_mask > 0]
        if len(surrounding_alpha) > 0:
            median_surrounding_alpha = np.median(surrounding_alpha)
            # If surrounding area is mostly transparent (median alpha < 128),
            # make inpainted areas transparent too
            if median_surrounding_alpha < 128:
                inpainted_mask = (mask > 127)
                result_alpha[inpainted_mask] = median_surrounding_alpha

    return result_alpha

class InpainterBase(BaseModule):

    inpaint_by_block = True
    check_need_inpaint = True

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in INPAINTERS.module_dict:
            if INPAINTERS.module_dict[key] == self.__class__:
                self.name = key
                break
    
    def memory_safe_inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        '''
        Handle cuda OOM (fallback to CPU). Tiling is disabled for all inpainters to prevent
        horizontal/vertical band and grid artifacts; use full-image mode or smaller inpaint_size if OOM.
        '''
        import torch

        def do_inpaint(im, msk, blk_list=None):
            return self._inpaint(im, msk, blk_list)

        try:
            return do_inpaint(img, mask, textblock_list)
        except Exception as e:
            if DEFAULT_DEVICE == 'cuda' and isinstance(e, torch.cuda.OutOfMemoryError):
                soft_empty_cache()
                try:
                    return self._inpaint(img, mask, textblock_list)
                except Exception as ee:
                    if isinstance(ee, torch.cuda.OutOfMemoryError):
                        self.logger.warning(f'CUDA out of memory while calling {self.name}, fall back to cpu...\n\
                                            if running into it frequently, consider lowering the inpaint_size')
                        self.moveToDevice('cpu')
                        inpainted = self._inpaint(img, mask, textblock_list)
                        precision = None
                        if hasattr(self, 'precision'):
                            precision = self.precision
                        self.moveToDevice('cuda', precision)

                        return inpainted
            else:
                raise e

    def inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None, check_need_inpaint: bool = False) -> np.ndarray:
        
        if not self.all_model_loaded():
            self.load_model()
        
        # Ensure mask and image dimensions match (e.g. mask loaded from file at different resolution)
        im_h, im_w = img.shape[:2]
        mask = np.copy(mask)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.shape[0] != im_h or mask.shape[1] != im_w:
            mask = cv2.resize(mask, (im_w, im_h), interpolation=cv2.INTER_NEAREST)
        
        # Binary mask: 255 = region to inpaint (hole); force strict 0/255 for C backends (OpenCV, PatchMatch)
        binary = (mask > 127).astype(np.uint8) * 255
        mask = np.where(binary > 0, 255, 0).astype(np.uint8)
        # Default 0: detector/refinement already dilate; extra dilation can over-expand and cause dark blobs for all models.
        # Subclasses (e.g. LamaLarge) can set mask_dilation_iterations if they need a small margin.
        dilate_iter = getattr(self, 'mask_dilation_iterations', 0)
        if dilate_iter > 0:
            k = getattr(self, 'mask_dilation_kernel_size', 2)
            k = max(2, min(5, int(k)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        mask = np.ascontiguousarray(mask)
        
        # Warn if mask is suspiciously large (often causes dark blobs with Lama; try opencv-telea or patchmatch, or re-run detection)
        mask_ratio = np.sum(mask > 127) / (im_h * im_w)
        if mask_ratio > 0.5:
            self.logger.warning(
                f'Inpaint mask covers {mask_ratio*100:.0f}% of the image. If you see dark blobs, try Config → Inpainting → opencv-telea or patchmatch, '
                'or reduce detector "mask dilate size" and re-run detection.'
            )
        
        # Handle RGBA images by preserving alpha channel
        original_alpha = None
        if len(img.shape) == 3 and img.shape[2] == 4:
            original_alpha = img[:, :, 3:4]  # Keep alpha channel
            img_rgb = img[:, :, :3]  # Use only RGB for inpainting
        else:
            img_rgb = img
        if mask_ratio <= 0:
            return np.concatenate([img_rgb, original_alpha], axis=2) if original_alpha is not None else np.ascontiguousarray(img_rgb.copy())
        
        if not self.inpaint_by_block or textblock_list is None:
            if check_need_inpaint:
                ballon_msk, non_text_msk = extract_ballon_mask(img_rgb, mask)
                if ballon_msk is not None:
                    non_text_region = np.where(non_text_msk > 0)
                    non_text_px = img_rgb[non_text_region]
                    average_bg_color = np.median(non_text_px, axis=0).astype(np.uint8)
                    std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                    std_max = np.max(std_rgb)
                    inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                    ballon_area = np.sum(ballon_msk > 0)
                    min_ballon_area_for_median = 40000
                    if std_max < inpaint_thresh and ballon_area >= min_ballon_area_for_median:
                        result_rgb = img_rgb.copy()
                        if np.all(average_bg_color >= 220):
                            average_bg_color = np.array([255, 255, 255], dtype=np.uint8)
                        result_rgb[np.where(ballon_msk > 0)] = average_bg_color
                        # Recombine with alpha if original was RGBA
                        if original_alpha is not None:
                            return np.concatenate([result_rgb, original_alpha], axis=2)
                        return result_rgb
            img_rgb = np.ascontiguousarray(img_rgb)
            result_rgb = self.memory_safe_inpaint(img_rgb, mask, textblock_list)
            # Recombine with alpha if original was RGBA
            if original_alpha is not None:
                result_alpha = inpaint_handle_alpha_channel(original_alpha, mask)
                return np.concatenate([result_rgb, result_alpha], axis=2)
            return result_rgb
        else:
            im_h, im_w = img_rgb.shape[:2]
            inpainted = np.copy(img_rgb)
            
            # Preserve original mask for transparency analysis
            original_mask = mask.copy()
            
            # Optional: exclude blocks by detector label (e.g. scene text); off by default so all blocks are inpainted
            if getattr(pcfg.module, 'inpaint_exclude_labels_enabled', False):
                exclude_str = (getattr(pcfg.module, 'inpaint_exclude_labels', None) or '').strip()
                if exclude_str:
                    excluded_labels = {s.strip().lower() for s in exclude_str.split(',') if s.strip()}
                    textblock_list = [b for b in textblock_list if (getattr(b, 'label', None) or '').strip().lower() not in excluded_labels]
            
            # Crop enlargement: use ratio 1.7 to match upstream (more context = better inpainting)
            enlarge_ratio = getattr(self, 'inpaint_enlarge_ratio', 1.7)
            for blk in textblock_list:
                xyxy = blk.xyxy
                xyxy_e = enlarge_window(xyxy, im_w, im_h, ratio=enlarge_ratio)
                # Skip blocks with degenerate crop (e.g. zero-area bbox or invalid enlarge_window result)
                crop_w = xyxy_e[2] - xyxy_e[0]
                crop_h = xyxy_e[3] - xyxy_e[1]
                if crop_w < 2 or crop_h < 2:
                    continue
                im = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                msk = mask[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                # Skip if this crop has no text mask left (e.g. already covered by a previous block's enlarged region)
                if np.sum(msk > 127) == 0:
                    continue
                need_inpaint = True
                if self.check_need_inpaint or check_need_inpaint:
                    ballon_msk, non_text_msk = extract_ballon_mask(im, msk)
                    if ballon_msk is not None:
                        non_text_region = np.where(non_text_msk > 0)
                        non_text_px = im[non_text_region]
                        average_bg_color = np.median(non_text_px, axis=0).astype(np.uint8)
                        std_rgb = np.std(non_text_px - average_bg_color, axis=0)
                        std_max = np.max(std_rgb)
                        inpaint_thresh = 7 if np.std(std_rgb) > 1 else 10
                        ballon_area = np.sum(ballon_msk > 0)
                        # Skip median fill for small balloons so they get proper inpainting
                        min_ballon_area_for_median = 40000  # ~200x200
                        if std_max < inpaint_thresh and ballon_area >= min_ballon_area_for_median:
                            need_inpaint = False
                            # Use pure white for speech bubbles when median is already near white
                            if np.all(average_bg_color >= 220):
                                average_bg_color = np.array([255, 255, 255], dtype=np.uint8)
                            im[np.where(ballon_msk > 0)] = average_bg_color
                        # cv2.imshow('im', im)
                        # cv2.imshow('ballon', ballon_msk)
                        # cv2.imshow('non_text', non_text_msk)
                        # cv2.waitKey(0)
                
                if need_inpaint:
                    result_crop = self.memory_safe_inpaint(im, msk)
                    # Ensure result matches crop size (some models return different size due to stride/pad)
                    ch, cw = im.shape[:2]
                    if result_crop.shape[0] != ch or result_crop.shape[1] != cw:
                        result_crop = cv2.resize(result_crop, (cw, ch), interpolation=cv2.INTER_LINEAR)
                    # Feather blend at crop edges to avoid visible rectangular seams (striped/grid artifacts)
                    feather_px = min(6, ch // 4, cw // 4)
                    if feather_px > 0:
                        w = _feather_weight_2d(ch, cw, feather_px)
                        if result_crop.ndim == 3:
                            w = w[:, :, np.newaxis]
                        roi = inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]]
                        blended = (w * result_crop.astype(np.float32) + (1 - w) * roi.astype(np.float32)).astype(np.uint8)
                        inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = blended
                    else:
                        inpainted[xyxy_e[1]:xyxy_e[3], xyxy_e[0]:xyxy_e[2]] = result_crop

                # Clear only block bbox in mask (match upstream); keeps margin in mask for adjacent blocks
                mask[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]] = 0
            
            # Recombine with alpha if original was RGBA
            if original_alpha is not None:
                result_alpha = inpaint_handle_alpha_channel(original_alpha, original_mask)
                return np.concatenate([inpainted, result_alpha], axis=2)
            return inpainted

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        raise NotImplementedError
    
    def moveToDevice(self, device: str, precision: str = None):
        raise not NotImplementedError


@register_inpainter('opencv-tela')
class OpenCVInpainter(InpainterBase):

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.inpaint_method = lambda img, mask, *args, **kwargs: cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True

    def is_cpu_intensive(self) -> bool:
        return True


@register_inpainter('opencv-telea')
class OpenCVTeleaInpainter(InpainterBase):
    """OpenCV Telea inpainting (#126). Fast, CPU-only, no model download."""

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.inpaint_method = lambda img, mask, *args, **kwargs: cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True

    def is_cpu_intensive(self) -> bool:
        return True
    


@register_inpainter('patchmatch')
class PatchmatchInpainter(InpainterBase):

    if sys.platform == 'darwin':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/macos_arm64_patchmatch_libs.7z',
                'sha256_pre_calculated': ['843704ab096d3afd8709abe2a2c525ce3a836bb0a629ed1ee9b8f5cee9938310', '849ca84759385d410c9587d69690e668822a3fc376ce2219e583e7e0be5b5e9a'],
                'files': ['macos_libopencv_world.4.8.0.dylib', 'macos_libpatchmatch_inpaint.dylib'],
                'save_dir': 'data/libs',
                'archived_files': 'macos_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': '9f332c888be0f160dbe9f6d6887eb698a302e62f4c102a0f24359c540d5858ea'
        }]
    elif sys.platform == 'win32':
        download_file_list = [{
                'url': 'https://github.com/dmMaze/PyPatchMatchInpaint/releases/download/v1.0/windows_patchmatch_libs.7z',
                'sha256_pre_calculated': ['3b7619caa29dc3352b939de4e9981217a9585a13a756e1101a50c90c100acd8d', '0ba60cfe664c97629daa7e4d05c0888ebfe3edcb3feaf1ed5a14544079c6d7af'],
                'files': ['opencv_world455.dll', 'patchmatch_inpaint.dll'],
                'save_dir': 'data/libs',
                'archived_files': 'windows_patchmatch_libs.7z',
                'archive_sha256_pre_calculated': 'c991ff61f7cb3efaf8e75d957e62d56ba646083bc25535f913ac65775c16ca65'
        }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        from . import patch_match
        # patch_size 3 is too small and can cause dark/blocky artifacts; use 7 for better quality (paper uses ~8-15)
        self.inpaint_method = lambda img, mask, *args, **kwargs: patch_match.inpaint(img, mask, patch_size=7)
    
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:
        return self.inpaint_method(img, mask)

    def is_computational_intensive(self) -> bool:
        return True
    
    def is_cpu_intensive(self) -> bool:
        return True


import torch
from utils.imgproc_utils import resize_keepasp
from .aot import AOTGenerator, load_aot_model


@register_inpainter('aot')
class AOTInpainter(InpainterBase):

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'value': 2048
        }, 
        'device': DEVICE_SELECTOR(),
        'description': 'manga-image-translator inpainter'
    }

    device = DEFAULT_DEVICE
    inpaint_size = 2048
    model: AOTGenerator = None
    _load_model_keys = {'model'}

    download_file_list = [{
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt',
            'sha256_pre_calculated': '878d541c68648969bc1b042a6e997f3a58e49b6c07c5636ad55130736977149f',
            'files': 'data/models/aot_inpainter.ckpt',
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.model: AOTGenerator = None
        
    def _load_model(self):
        AOTMODEL_PATH = 'data/models/aot_inpainter.ckpt'
        self.model = load_aot_model(AOTMODEL_PATH, self.device)

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None

        img = resize_keepasp(img, new_shape, stride=None)
        mask = resize_keepasp(mask, new_shape, stride=None)

        im_h, im_w = img.shape[:2]
        pad_bottom = 128 - im_h if im_h < 128 else 0
        pad_right = 128 - im_w if im_w < 128 else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 127.5 - 1.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        img_inpainted_torch = self.model(img_torch, mask_torch)
        img_inpainted = ((img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() + 1.0) * 127.5)
        img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])


from .lama import LamaFourier, load_lama_mpe

@register_inpainter('lama_mpe')
class LamaInpainterMPE(InpainterBase):

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                1024, 
                2048
            ], 
            'value': 2048
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone'])
    }

    download_file_list = [{
            'url': 'https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt',
            'sha256_pre_calculated': 'd625aa1b3e0d0408acfd6928aa84f005867aa8dbb9162480346a4e20660786cc',
            'files': 'data/models/lama_mpe.ckpt',
    }]
    _load_model_keys = {'model'}

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.device = self.params['device']['value']
        self.inpaint_size = int(self.params['inpaint_size']['value'])
        self.precision = 'fp32'
        self.model: LamaFourier = None

    def _load_model(self):
        self.model = load_lama_mpe(r'data/models/lama_mpe.ckpt', self.device)

    def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_original = np.copy(img)
        mask_original = np.copy(mask)
        mask_original[mask_original < 127] = 0
        mask_original[mask_original >= 127] = 1
        mask_original = mask_original[:, :, None]

        max_side = max(img.shape[0:2])
        if max_side > self.inpaint_size:
            new_shape = self.inpaint_size
        elif max_side < 400:
            # Small bubbles: normalize to 512 max to reduce over-strong inpainting and artifacts
            new_shape = min(self.inpaint_size, 512)
        else:
            new_shape = None
        # high resolution input could produce cloudy artifacts
        img = resize_keepasp(img, new_shape, stride=64)
        mask = resize_keepasp(mask, new_shape, stride=64)

        im_h, im_w = img.shape[:2]
        longer = max(im_h, im_w)
        pad_bottom = longer - im_h if im_h < longer else 0
        pad_right = longer - im_w if im_w < longer else 0
        mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
        img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

        img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
        mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
        mask_torch[mask_torch < 0.5] = 0
        mask_torch[mask_torch >= 0.5] = 1
        rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
        rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
        direct = torch.LongTensor(direct).unsqueeze_(0)

        if self.device != 'cpu':
            img_torch = img_torch.to(self.device)
            mask_torch = mask_torch.to(self.device)
            rel_pos = rel_pos.to(self.device)
            direct = direct.to(self.device)
        img_torch *= (1 - mask_torch)
        return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

    @torch.no_grad()
    def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

        im_h, im_w = img.shape[:2]
        img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
        
        precision = TORCH_DTYPE_MAP[self.precision]
        if self.device in {'cuda'}:
            try:
                with torch.autocast(device_type=self.device, dtype=precision):
                    img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
            except Exception as e:
                self.logger.error(e)
                self.logger.error(f'{precision} inference is not supported for this device, use fp32 instead.')
                img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        else:
            img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)

        img_inpainted = (img_inpainted_torch.to(device='cpu', dtype=torch.float32).squeeze_(0).permute(1, 2, 0).numpy() * 255)
        img_inpainted = (np.clip(np.round(img_inpainted), 0, 255)).astype(np.uint8)
        if pad_bottom > 0:
            img_inpainted = img_inpainted[:-pad_bottom]
        if pad_right > 0:
            img_inpainted = img_inpainted[:, :-pad_right]
        new_shape = img_inpainted.shape[:2]
        if new_shape[0] != im_h or new_shape[1] != im_w :
            img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
        img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
        return img_inpainted

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)

        if param_key == 'device':
            param_device = self.params['device']['value']
            if self.model is not None:
                self.model.to(param_device)
            self.device = param_device

        elif param_key == 'inpaint_size':
            self.inpaint_size = int(self.params['inpaint_size']['value'])

        elif param_key == 'precision':
            precision = self.params['precision']['value']
            self.precision = precision

    def moveToDevice(self, device: str, precision: str = None):
        self.model.to(device)
        self.device = device
        if precision is not None:
            self.precision = precision

@register_inpainter('lama_large_512px')
class LamaLarge(LamaInpainterMPE):

    mask_dilation_iterations = 1
    mask_dilation_kernel_size = 2  # 2×2 = gentler expansion per iteration; 3×3 = stronger
    # Always run LaMa; skip median fill to avoid "weird box of a certain color" in speech bubbles
    check_need_inpaint = False

    params = {
        'inpaint_size': {
            'type': 'selector',
            'options': [
                512,
                768,
                1024,
                1536,
                2048
            ],
            'value': 1024,
        },
        'mask_dilation': {
            'type': 'selector',
            'options': [0, 1, 2, 3, 4, 5],
            'value': 1,
            'description': 'Mask dilation iterations (0–5). 0 = no expansion, keeps bubble edges. 1 = minimal. Higher values expand the mask more and can erase bubble shape.',
        },
        'mask_dilation_kernel': {
            'type': 'selector',
            'options': [2, 3],
            'value': 2,
            'description': 'Dilation kernel size (2 or 3). 2 = gentler expansion per iteration; 3 = stronger. Use 2 if inpainting is too strong.',
        },
        'inpaint_enlarge_ratio': {
            'type': 'selector',
            'options': [1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.7],
            'value': 1.7,
            'description': 'Crop margin ratio (1.1–1.7). 1.7 = upstream default, more context for inpainting. Lower = tighter crop.',
        },
        'device': DEVICE_SELECTOR(not_supported=['privateuseone']),
        'precision': {
            'type': 'selector',
            'options': [
                'fp32',
                'bf16'
            ],
            'value': 'bf16' if BF16_SUPPORTED == 'cuda' else 'fp32'
        },
    }

    download_file_list = [{
            'url': 'https://huggingface.co/dreMaz/AnimeMangaInpainting/resolve/main/lama_large_512px.ckpt',
            'sha256_pre_calculated': '11d30fbb3000fb2eceae318b75d9ced9229d99ae990a7f8b3ac35c8d31f2c935',
            'files': 'data/models/lama_large_512px.ckpt',
    }]

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self.precision = self.params['precision']['value']
        self._update_mask_dilation()
        self._update_mask_dilation_kernel()

    def _update_mask_dilation(self):
        self.mask_dilation_iterations = int(self.params.get('mask_dilation', {}).get('value', 1))

    def _update_mask_dilation_kernel(self):
        self.mask_dilation_kernel_size = int(self.params.get('mask_dilation_kernel', {}).get('value', 2))

    def _load_model(self):
        device = self.params['device']['value']
        precision = self.params['precision']['value']

        self.model = load_lama_mpe(r'data/models/lama_large_512px.ckpt', device='cpu', use_mpe=False, large_arch=True)
        self.moveToDevice(device, precision=precision)

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'mask_dilation':
            self._update_mask_dilation()
        elif param_key == 'mask_dilation_kernel':
            self._update_mask_dilation_kernel()


# LAMA_ORI: LamaFourier = None
# @register_inpainter('lama_ori')
# class LamaInpainterORI(InpainterBase):

#     params = {
#         'inpaint_size': {
#             'type': 'selector',
#             'options': [
#                 1024, 
#                 2048
#             ], 
#             'value': 2048
#         }, 
#         'device': {
#             'type': 'selector',
#             'options': [
#                 'cpu',
#                 'cuda'
#             ],
#             'value': DEFAULT_DEVICE
#         }
#     }

#     device = DEFAULT_DEVICE
#     inpaint_size = 2048

#     def setup_inpainter(self):
#         global LAMA_ORI

#         self.device = self.params['device']['value']
#         if LAMA_ORI is None:
#             self.model = LAMA_ORI = load_lama_mpe(r'data/models/lama_org.ckpt', self.device, False)
#         else:
#             self.model = LAMA_ORI
#             self.model.to(self.device)
#         self.inpaint_by_block = True if self.device == 'cuda' else False
#         self.inpaint_size = int(self.params['inpaint_size']['value'])

#     def inpaint_preprocess(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:

#         img_original = np.copy(img)
#         mask_original = np.copy(mask)
#         mask_original[mask_original < 127] = 0
#         mask_original[mask_original >= 127] = 1
#         mask_original = mask_original[:, :, None]

#         new_shape = self.inpaint_size if max(img.shape[0: 2]) > self.inpaint_size else None
#         # high resolution input could produce cloudy artifacts
#         img = resize_keepasp(img, new_shape, stride=64)
#         mask = resize_keepasp(mask, new_shape, stride=64)

#         im_h, im_w = img.shape[:2]
#         longer = max(im_h, im_w)
#         pad_bottom = longer - im_h if im_h < longer else 0
#         pad_right = longer - im_w if im_w < longer else 0
#         mask = cv2.copyMakeBorder(mask, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)
#         img = cv2.copyMakeBorder(img, 0, pad_bottom, 0, pad_right, cv2.BORDER_REFLECT)

#         img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze_(0).float() / 255.0
#         mask_torch = torch.from_numpy(mask).unsqueeze_(0).unsqueeze_(0).float() / 255.0
#         mask_torch[mask_torch < 0.5] = 0
#         mask_torch[mask_torch >= 0.5] = 1
#         rel_pos, _, direct = self.model.load_masked_position_encoding(mask_torch[0][0].numpy())
#         rel_pos = torch.LongTensor(rel_pos).unsqueeze_(0)
#         direct = torch.LongTensor(direct).unsqueeze_(0)

#         if self.device == 'cuda':
#             img_torch = img_torch.cuda()
#             mask_torch = mask_torch.cuda()
#             rel_pos = rel_pos.cuda()
#             direct = direct.cuda()
#         img_torch *= (1 - mask_torch)
#         return img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right

#     @torch.no_grad()
#     def _inpaint(self, img: np.ndarray, mask: np.ndarray, textblock_list: List[TextBlock] = None) -> np.ndarray:

#         im_h, im_w = img.shape[:2]
#         img_torch, mask_torch, rel_pos, direct, img_original, mask_original, pad_bottom, pad_right = self.inpaint_preprocess(img, mask)
#         img_inpainted_torch = self.model(img_torch, mask_torch, rel_pos, direct)
        
#         img_inpainted = (img_inpainted_torch.cpu().squeeze_(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
#         if pad_bottom > 0:
#             img_inpainted = img_inpainted[:-pad_bottom]
#         if pad_right > 0:
#             img_inpainted = img_inpainted[:, :-pad_right]
#         new_shape = img_inpainted.shape[:2]
#         if new_shape[0] != im_h or new_shape[1] != im_w :
#             img_inpainted = cv2.resize(img_inpainted, (im_w, im_h), interpolation = cv2.INTER_LINEAR)
#         img_inpainted = img_inpainted * mask_original + img_original * (1 - mask_original)
        
#         return img_inpainted

#     def updateParam(self, param_key: str, param_content):
#         super().updateParam(param_key, param_content)

#         if param_key == 'device':
#             param_device = self.params['device']['value']
#             self.model.to(param_device)
#             self.device = param_device
#             if param_device == 'cuda':
#                 self.inpaint_by_block = False
#             else:
#                 self.inpaint_by_block = True

#         elif param_key == 'inpaint_size':
#             self.inpaint_size = int(self.params['inpaint_size']['value'])