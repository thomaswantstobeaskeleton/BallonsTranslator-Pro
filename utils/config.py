import json, os, traceback
import os.path as osp
import copy
from typing import Callable, Optional

from . import shared
from .fontformat import FontFormat
from .structures import List, Dict, Config, field, nested_dataclass
from .logger import logger as LOGGER
from .io_utils import json_dump_nested_obj, np, serialize_np

class RunStatus:
    FIN_DET = 1
    FIN_OCR = 2
    FIN_INPAINT = 4
    FIN_TRANSLATE = 8
    FIN_ALL = 15


@nested_dataclass
class ModuleConfig(Config):
    textdetector: str = 'ctd'
    ocr: str = "mit48px"
    inpainter: str = 'lama_large_512px'
    translator: str = "google"
    enable_detect: bool = True
    enable_dual_detect: bool = False
    textdetector_secondary: str = ''
    # When True, only add secondary (and tertiary) detector boxes that are outside primary bubbles (no in-bubble duplicates).
    secondary_detector_outside_bubble_only: bool = False
    enable_tertiary_detect: bool = False
    textdetector_tertiary: str = ''
    keep_exist_textlines: bool = False
    enable_ocr: bool = True
    enable_translate: bool = True
    enable_inpaint: bool = True
    # 是否在 OCR 后进行字体检测（默认不启用）
    ocr_font_detect: bool = False
    textdetector_params: Dict = field(default_factory=lambda: dict())
    ocr_params: Dict = field(default_factory=lambda: dict())
    translator_params: Dict = field(default_factory=lambda: dict())
    inpainter_params: Dict = field(default_factory=lambda: dict())
    translate_source: str = '日本語'
    translate_target: str = '简体中文'
    check_need_inpaint: bool = True
    inpaint_tile_size: int = 0   # 0 = no tiling (recommended); set 512–1024 only if OOM
    inpaint_tile_overlap: int = 64   # overlap between tiles (px); only used when tile_size > 0
    # Optional: exclude text blocks by detector label from inpainting (e.g. leave scene text as-is). Off by default.
    inpaint_exclude_labels_enabled: bool = False
    inpaint_exclude_labels: str = ''  # comma-separated, case-insensitive (e.g. "other,scene")
    # When True, inpaint the whole image at once (no per-block crops). Uses more VRAM/slower but avoids per-block issues; try if Lama gives bad results.
    inpaint_full_image: bool = False
    load_model_on_demand: bool = False
    empty_runcache: bool = False
    # Optional: panel-aware reading order for block sorting (affects translation prompt order & typesetting sequence).
    enable_panel_order: bool = False
    # "auto" uses heuristic based on detected text orientation; "rtl" forces right-to-left; "ltr" forces left-to-right.
    panel_reading_direction: str = "auto"
    # Outside-speech-bubble (OSB) / text_free pipeline (optional, only works when detector sets blk.label, e.g. HF object det).
    enable_osb_pipeline: bool = False
    # Group nearby OSB boxes into larger regions (captions/SFX clusters).
    osb_group_nearby: bool = True
    osb_group_gap_px: int = 24
    # Drop OSB boxes that overlap bubble boxes by IoU (to avoid double-processing text inside bubbles).
    osb_exclude_bubble_iou: float = 0.10
    # After OCR, remove small margin page-number-like OSB blocks (e.g. "12"). Requires OCR enabled.
    osb_page_number_filter: bool = False
    osb_page_number_margin_ratio: float = 0.08
    # Probe OSB background and set readable fg/stroke defaults for rendering (does not override user styles later).
    osb_style_probe: bool = False
    # When True, OSB regions may be filled with median surrounding color if background is low-variance (faster/safer).
    osb_fast_fill: bool = False
    # Section 14: expand bubble boxes to fully contain overlapping OSB text so mask includes all text pixels.
    osb_expand_bubbles_with_osb: bool = True
    # Section 19: when OSB layout fails, retry with vertical stacking then restore original crop. Disable to only set restore_original_region on first failure.
    osb_layout_fallbacks_enabled: bool = True
    # Section 15: resolve overlapping mask regions by bisector split (and nudge for text boxes).
    resolve_mask_overlaps_bisector: bool = True
    # Section 16: cleaning quality — adaptive shrink at conjoined junctions, Otsu retry on failure.
    cleaning_adaptive_shrink_junction: bool = True
    cleaning_otsu_retry: bool = True
    # Section 17: colored/gradient bubbles — classify and inpaint text-only; re-sample brightness for contrast.
    colored_bubble_handling: bool = True
    colored_bubble_resample_brightness: bool = True
    # Translation workflow
    # - two_step: OCR -> text translator (default)
    # - one_step_vlm: OCR module performs translate-from-image and writes blk.translation directly (requires supported OCR, e.g. llm_ocr)
    translation_mode: str = "two_step"
    # Replace translation mode (manga-translator-ui style): input = raw image + pre-rendered translated image (same name in a folder).
    # Pipeline: detect+OCR on translated image -> text; detect on raw -> blocks; match by position; inpaint raw; render matched text.
    replace_translation_mode: bool = False
    replace_translation_translated_dir: str = ""  # Folder containing translated images (same filenames as project). Empty = off.
    # When True (default), on soft translation failure (e.g. timeout, parse error) use placeholder and continue batch. When False, show dialog and stop like critical errors.
    translation_soft_failure_continue: bool = True
    # Skip translation for pages that already have all blocks translated (non-empty translation). Speeds up re-runs.
    skip_already_translated: bool = False
    # Optional: merge nearby text blocks using collision-based grouping (Dango-style). Off by default; enable for word-level OCR or many small blocks.
    merge_nearby_blocks_collision: bool = False
    merge_nearby_blocks_gap_ratio: float = 1.5  # vertical expansion ratio for horizontal merge; 1.5 = Dango-style
    # Only run collision merge when page has at least this many blocks (avoids merging normal bubble layouts; default 18).
    merge_nearby_blocks_min_blocks: int = 18
    # Translation caching (saves API costs for deterministic settings/reruns)
    translation_cache_enabled: bool = False
    translation_cache_deterministic_only: bool = True
    # Typesetting / layout (auto layout for translated text blocks)
    layout_optimal_breaks: bool = True
    layout_hyphenation: bool = True
    layout_collision_check: bool = True
    layout_collision_min_mask_ratio: float = 0.85
    layout_collision_max_retries: int = 3
    # Center text vertically (and horizontally as needed) inside each bubble/block (manga-translator-ui style).
    center_text_in_bubble: bool = False
    # Try larger font / fewer lines so text fits with fewer line breaks (test combinations; manga-translator-ui style).
    optimize_line_breaks: bool = False
    finish_code: int = 15
    run_preset_name: str = 'Full'
    # --- Section 6 / 6.1: Image upscaling & per-stage resizing ---
    # Global default min side for OCR crop upscale (0 = off). Per-OCR params (e.g. EasyOCR upscale_min_side) override when set.
    ocr_upscale_min_side: int = 0
    # Initial upscale: before detection/OCR (improves small text). Pipeline runs on upscaled image then downscales results to original size.
    image_upscale_initial: bool = False
    image_upscale_initial_factor: float = 2.0
    # Final output upscale: 2x (or factor) when saving result image.
    image_upscale_final: bool = False
    image_upscale_final_factor: float = 2.0
    # Auto-scale pipeline params by image area (processing_scale = sqrt(area/1e6)) for fonts, padding, morphology, thresholds.
    processing_scale_enabled: bool = True
    # Per-stage resize policy: none | lanczos (model/model_lite reserved for future).
    upscale_policy_initial: str = "lanczos"
    upscale_policy_final: str = "lanczos"
    # Section 7: Caching + memory / stability
    pipeline_cache_enabled: bool = False  # When True, in-memory pipeline cache can be used (get_pipeline_cache(True))
    inpaint_spill_to_disk_after_blocks: int = 0  # When >0, write intermediate inpainted image to temp file every N blocks to reduce peak RAM/VRAM (e.g. 8 or 12)

    def get_params(self, module_key: str, for_saving=False) -> dict:
        d = self[module_key + '_params']
        if not for_saving:
            return d
        sd = {}
        for module_key, module_params in d.items():
            if module_params is None:
                continue
            saving_module_params = {}
            sd[module_key] = saving_module_params
            for pk, pv in module_params.items():
                if pk in {'description'}:
                    continue
                if pk.startswith('__'):
                    continue
                if isinstance(pv, dict):
                    pv = pv['value']
                saving_module_params[pk] = pv
        return sd

    def get_saving_params(self, to_dict=True):
        params = copy.copy(self)
        params.ocr_params = self.get_params('ocr', for_saving=True)
        params.inpainter_params = self.get_params('inpainter', for_saving=True)
        params.textdetector_params = self.get_params('textdetector', for_saving=True)
        params.translator_params = self.get_params('translator', for_saving=True)
        if to_dict:
            return params.__dict__
        return params
    
    def stage_enabled(self, idx: int):
        if idx == 0:
            return self.enable_detect
        elif idx == 1:
            return self.enable_ocr
        elif idx == 2:
            return self.enable_translate
        elif idx == 3:
            return self.enable_inpaint
        else:
            raise Exception(f'not supported stage idx: {idx}')
        
    def all_stages_disabled(self):
        return (self.enable_detect or self.enable_ocr or self.enable_translate or self.enable_inpaint) is False

    def __post_init__(self):
        self.update_finish_code()

    def update_finish_code(self):
        self.finish_code = self.enable_detect * RunStatus.FIN_DET + \
            self.enable_ocr * RunStatus.FIN_OCR + \
                self.enable_translate * RunStatus.FIN_TRANSLATE + \
                    self.enable_inpaint * RunStatus.FIN_INPAINT
        

@nested_dataclass
class DrawPanelConfig(Config):
    pentool_color: List = field(default_factory=lambda: [0, 0, 0, 255])  # [r, g, b, a]
    pentool_width: float = 30.
    pentool_shape: int = 0
    inpainter_width: float = 30.
    inpainter_shape: int = 0
    inpaint_hardness: int = 100  # 100 = hard edge, 0 = soft/feathered
    current_tool: int = 0
    rectool_auto: bool = False
    rectool_method: int = 0
    rectool_shape: int = 0  # 0 = Rectangle, 1 = Ellipse (#35)
    recttool_dilate_ksize: int = 0
    recttool_erode_ksize: int = 0
    # Optional: SAM2/SAM3 refinement for balloon masks (used by the inpaint mask-seg method).
    # Requires transformers with SAM2/SAM3 support; if unavailable, the app falls back gracefully.
    sam_maskrefine_model_id: str = "facebook/sam2.1-hiera-large"
    # Empty => auto-select ("cuda" if available else "cpu"). You can also set "cuda" / "cpu".
    sam_maskrefine_device: str = ""
    # Expand the prompt box around the coarse mask by this many pixels (crop-local coords).
    sam_maskrefine_padding_px: int = 12

@nested_dataclass
class ProgramConfig(Config):

    module: ModuleConfig = field(default_factory=lambda: ModuleConfig())
    drawpanel: DrawPanelConfig = field(default_factory=lambda: DrawPanelConfig())
    global_fontformat: FontFormat = field(default_factory=lambda: FontFormat())
    recent_proj_list: List = field(default_factory=lambda: list())
    show_page_list: bool = False
    imgtrans_paintmode: bool = False
    imgtrans_textedit: bool = True
    imgtrans_textblock: bool = True
    mask_transparency: float = 0.
    original_transparency: float = 0.
    open_recent_on_startup: bool = True
    recent_proj_list_max: int = 14
    # When True, show the welcome screen on startup when no project is opened (manhua-translator / Komakun style).
    show_welcome_screen: bool = True
    # When True, check for and pull GitHub updates on startup. Can cause issues or bad results; use with caution.
    auto_update_from_github: bool = False
    logical_dpi: int = 0
    confirm_before_run: bool = True
    let_fntsize_flag: int = 0
    let_fntstroke_flag: int = 0
    let_fntcolor_flag: int = 0
    let_fnt_scolor_flag: int = 0
    let_fnteffect_flag: int = 1
    let_alignment_flag: int = 0
    let_writing_mode_flag: int = 0
    let_family_flag: int = 0
    let_autolayout_flag: bool = True
    let_uppercase_flag: bool = True
    let_show_only_custom_fonts_flag: bool = False
    let_textstyle_indep_flag: bool = False
    text_styles_path: str = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')
    fsearch_case: bool = False
    fsearch_whole_word: bool = False
    fsearch_regex: bool = False
    fsearch_range: int = 0
    gsearch_case: bool = False
    gsearch_whole_word: bool = False
    gsearch_regex: bool = False
    gsearch_range: int = 0
    darkmode: bool = False
    bubbly_ui: bool = True
    accent_color_hex: str = ''  # Theme customizer: e.g. #1E93E5 (blue) or #9B59B6 (purple). Empty = use theme default.
    app_font_family: str = ''   # Theme customizer: app-wide font. Empty = system default.
    app_font_size: int = 0      # Theme customizer: app-wide font size. 0 = system default.
    use_custom_cursor: bool = False
    custom_cursor_path: str = ''
    textselect_mini_menu: bool = True
    fold_textarea: bool = False
    show_source_text: bool = True
    show_trans_text: bool = True
    saladict_shortcut: str = "Alt+S"
    search_url: str = "https://www.google.com/search?q="
    ocr_sublist: List = field(default_factory=lambda: list())
    restore_ocr_empty: bool = False
    pre_mt_sublist: List = field(default_factory=lambda: list())
    mt_sublist: List = field(default_factory=lambda: list())
    display_lang: str = field(default_factory=lambda: shared.DEFAULT_DISPLAY_LANG) # to always apply shared.DEFAULT_DISPLAY_LANG
    imgsave_quality: int = 100
    imgsave_webp_lossless: bool = False
    imgsave_ext: str = '.png'
    intermediate_imgsave_ext: str = '.png'
    supersampling_factor: int = 1  # 1 = off, 2..4 render at Nx then downscale for smoother edges
    # Section 10: Canvas view mode for QA (original / debug boxes-masks / translated / normal).
    canvas_view_mode: str = "normal"  # "normal" | "original" | "debug" | "translated"
    show_text_style_preset: bool = True
    expand_tstyle_panel: bool = True
    show_text_effect_panel: bool = True
    expand_teffect_panel: bool = True
    text_advanced_format_panel: bool = True
    expand_tadvanced_panel: bool = True
    config_panel_font_scale: float = 1.0
    default_device: str = ''
    unload_after_idle_minutes: int = 0
    ocr_spell_check: bool = False
    manga_source_lang: str = 'en'
    manga_source_data_saver: bool = False
    manga_source_download_dir: str = ''
    manga_source_request_delay: float = 0.3
    manga_source_open_after_download: bool = False
    manga_source_playwright_headless: bool = True
    manga_source_translate_raw_search: bool = True  # For raw sources: translate search query to Japanese/Korean/Chinese
    # Model packages to download at startup (None = legacy "all"; ["core"] = minimal). See utils.model_packages.
    model_packages_enabled: Optional[List[str]] = field(default_factory=lambda: ["core"])
    # When True, show all modules in detector/OCR/translator dropdowns (including not downloaded or incompatible). When False, only show ready modules.
    dev_mode: bool = False
    shortcuts: Dict = field(default_factory=dict)
    auto_region_merge_after_run: str = 'never'  # 'never' | 'all_pages' | 'current_page'
    region_merge_settings: Dict = field(default_factory=dict)  # Region merge tool dialog (persisted)
    context_menu: Dict = field(default_factory=dict)  # Canvas right-click: action key -> visible (default True)
    huggingface_token: str = ''  # Optional: gated models + faster HF downloads (Xet). Prefer env HF_TOKEN to avoid storing in config.
    translator_last_model_by_provider: Dict = field(default_factory=dict)  # Section 9: last-used model per LLM provider

    @staticmethod
    def default_downloaded_chapters_dir() -> str:
        """Return the default folder for downloaded chapters; creates it if it does not exist."""
        path = osp.join(osp.expanduser("~"), "BallonsTranslator", "Downloaded Chapters")
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as e:
            LOGGER.warning("Could not create default download folder %s: %s", path, e)
        return path

    @staticmethod
    def load(cfg_path: str):
        
        with open(cfg_path, 'r', encoding='utf8') as f:
            config_dict = json.loads(f.read())

        # for backward compatibility
        if 'dl' in config_dict:
            dl = config_dict.pop('dl')
            if not 'module' in config_dict:
                if 'textdetector_setup_params' in dl:
                    textdetector_params = dl.pop('textdetector_setup_params')
                    dl['textdetector_params'] = textdetector_params
                if 'inpainter_setup_params' in dl:
                    inpainter_params = dl.pop('inpainter_setup_params')
                    dl['inpainter_params'] = inpainter_params
                if 'ocr_setup_params' in dl:
                    ocr_params = dl.pop('ocr_setup_params')
                    dl['ocr_params'] = ocr_params
                if 'translator_setup_params' in dl:
                    translator_params = dl.pop('translator_setup_params')
                    dl['translator_params'] = translator_params
                config_dict['module'] = dl

        if 'module' in config_dict:
            module_cfg = config_dict['module']
            trans_params = module_cfg['translator_params']
            repl_pairs = {'baidu': 'Baidu', 'caiyun': 'Caiyun', 'chatgpt': 'ChatGPT', 'Deepl': 'DeepL', 'papago': 'Papago'}
            for k, i in repl_pairs.items():
                if k in trans_params:
                    trans_params[i] = trans_params.pop(k)
            if module_cfg['translator'] in repl_pairs:
                module_cfg['translator'] = repl_pairs[module_cfg['translator']]

        # Legacy: existing configs without this key used to download all models at startup
        config_dict.setdefault("model_packages_enabled", None)

        return ProgramConfig(**config_dict)
    

pcfg = ProgramConfig()
text_styles: List[FontFormat] = []
active_format: FontFormat = None

# Default keys for canvas context menu visibility (all True = show).
CONTEXT_MENU_DEFAULT = {
    'edit_copy': True, 'edit_paste': True, 'edit_copy_trans': True, 'edit_paste_trans': True,
    'edit_copy_src': True, 'edit_paste_src': True, 'edit_delete': True, 'edit_delete_recover': True,
    'edit_clear_src': True, 'edit_clear_trans': True, 'edit_select_all': True,
    'text_spell_src': True, 'text_spell_trans': True, 'text_trim': True, 'text_upper': True, 'text_lower': True,
    'text_strikethrough': True, 'text_gradient': True, 'text_on_path': True,
    'block_merge': True, 'block_split': True, 'block_move_up': True, 'block_move_down': True,
    'create_textbox': True,
    'overlay_import': True, 'overlay_clear': True,
    'transform_free': True, 'transform_reset_warp': True, 'transform_warp_preset': True,
    'order_bring_front': True, 'order_send_back': True,
    'format_apply': True, 'format_layout': True, 'format_auto_fit': True, 'format_angle': True, 'format_squeeze': True,
    'run_detect_region': True, 'run_detect_page': True, 'run_translate': True, 'run_ocr': True,
    'run_ocr_translate': True, 'run_ocr_translate_inpaint': True, 'run_inpaint': True,
}

# Section 9: canonical key order when saving config (clean diffs, easier debugging)
CONFIG_KEY_ORDER = (
    "module", "drawpanel", "global_fontformat", "recent_proj_list", "show_page_list",
    "imgtrans_paintmode", "imgtrans_textedit", "imgtrans_textblock", "mask_transparency", "original_transparency",
    "open_recent_on_startup", "recent_proj_list_max", "show_welcome_screen", "auto_update_from_github", "logical_dpi", "confirm_before_run",
    "let_fntsize_flag", "let_fntstroke_flag", "let_fntcolor_flag", "let_fnt_scolor_flag", "let_fnteffect_flag",
    "let_alignment_flag", "let_writing_mode_flag", "let_family_flag", "let_autolayout_flag", "let_uppercase_flag",
    "let_show_only_custom_fonts_flag", "let_textstyle_indep_flag", "text_styles_path",
    "fsearch_case", "fsearch_whole_word", "fsearch_regex", "fsearch_range",
    "gsearch_case", "gsearch_whole_word", "gsearch_regex", "gsearch_range",
    "darkmode", "bubbly_ui", "accent_color_hex", "app_font_family", "app_font_size", "use_custom_cursor", "custom_cursor_path", "textselect_mini_menu", "fold_textarea", "show_source_text", "show_trans_text",
    "saladict_shortcut", "search_url", "ocr_sublist", "restore_ocr_empty", "pre_mt_sublist", "mt_sublist",
    "display_lang", "imgsave_quality", "imgsave_webp_lossless", "imgsave_ext", "intermediate_imgsave_ext",
    "supersampling_factor", "show_text_style_preset", "expand_tstyle_panel", "show_text_effect_panel",
    "expand_teffect_panel", "text_advanced_format_panel", "expand_tadvanced_panel", "config_panel_font_scale",
    "default_device", "unload_after_idle_minutes", "ocr_spell_check",
    "manga_source_lang", "manga_source_data_saver", "manga_source_download_dir",
    "manga_source_request_delay", "manga_source_open_after_download", "manga_source_playwright_headless",
    "manga_source_translate_raw_search",
    "model_packages_enabled",
    "dev_mode",
    "shortcuts", "auto_region_merge_after_run", "region_merge_settings", "context_menu",
    "huggingface_token", "translator_last_model_by_provider",
)

def context_menu_visible(key: str) -> bool:
    """Whether the context menu action with this key should be shown. Missing key => True."""
    if not hasattr(pcfg, 'context_menu') or not isinstance(pcfg.context_menu, dict):
        return True
    return pcfg.context_menu.get(key, True)


def load_textstyle_from(p: str, raise_exception = False):

    if not osp.exists(p):
        LOGGER.warning(f'Text style {p} does not exist.')
        return

    try:
        with open(p, 'r', encoding='utf8') as f:
            style_list = json.loads(f.read())
            styles_loaded = []
            for style in style_list:
                try:
                    styles_loaded.append(FontFormat(**style))
                except Exception as e:
                    LOGGER.warning(f'Skip invalid text style: {style}')
    except Exception as e:
        LOGGER.error(f'Failed to load text style from {p}: {e}')
        if raise_exception:
            raise e
        return

    global text_styles, pcfg
    if len(text_styles) > 0:
        text_styles.clear()
    text_styles.extend(styles_loaded)
    pcfg.text_styles_path = p

def load_config(config_path: str = shared.CONFIG_PATH):
    if config_path != shared.CONFIG_PATH:
        shared.CONFIG_PATH = config_path
        LOGGER.info(f'Using specified config file at {shared.CONFIG_PATH}')

    config_file_existed = osp.exists(shared.CONFIG_PATH)
    if config_file_existed:
        try:
            config = ProgramConfig.load(shared.CONFIG_PATH)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            config = ProgramConfig()
        shared.FIRST_RUN_NO_CONFIG = False
    if not config_file_existed:
        LOGGER.info(f'{shared.CONFIG_PATH} does not exist, new config file will be created.')
        shared.FIRST_RUN_NO_CONFIG = True
        example_path = osp.join(osp.dirname(shared.CONFIG_PATH), 'config.example.json')
        if osp.isfile(example_path):
            try:
                config = ProgramConfig.load(example_path)
                LOGGER.info(f'Loaded recommended defaults from {example_path}.')
            except Exception as e:
                LOGGER.warning(f'Could not load config.example.json: {e}. Using code defaults.')
                config = ProgramConfig()
        else:
            config = ProgramConfig()
    
    global pcfg
    pcfg.merge(config)
    # Migrate removed rtdetr_comic / legacy rtdetr_v2 -> ctd (rtdetr_comic detector removed)
    for key in ('textdetector', 'textdetector_secondary', 'textdetector_tertiary'):
        if getattr(pcfg.module, key, None) in ('rtdetr_comic', 'rtdetr_v2'):
            setattr(pcfg.module, key, 'ctd')
    tp = getattr(pcfg.module, 'textdetector_params', None)
    if isinstance(tp, dict):
        for old in ('rtdetr_v2', 'rtdetr_comic'):
            tp.pop(old, None)
    # Section 9: clamp numeric settings
    try:
        from utils.validation import clamp_settings
        clamp_settings(pcfg)
    except Exception:
        pass
    # Merge context menu visibility with defaults (new keys default to True)
    if hasattr(pcfg, 'context_menu') and isinstance(pcfg.context_menu, dict):
        for k, v in CONTEXT_MENU_DEFAULT.items():
            if k not in pcfg.context_menu:
                pcfg.context_menu[k] = v
    else:
        pcfg.context_menu = dict(CONTEXT_MENU_DEFAULT)
    # Ensure all shortcut keys exist (merge defaults for any new action)
    try:
        from utils.shortcuts import get_default_shortcuts
        defaults = get_default_shortcuts()
        if not isinstance(getattr(pcfg, 'shortcuts', None), dict):
            pcfg.shortcuts = dict(defaults)
        else:
            for k, v in defaults.items():
                if k not in pcfg.shortcuts:
                    pcfg.shortcuts[k] = v
    except Exception:
        pass
    # Trim recent projects to configured max
    max_recent = getattr(pcfg, 'recent_proj_list_max', 14)
    if max_recent > 0 and len(pcfg.recent_proj_list) > max_recent:
        pcfg.recent_proj_list = pcfg.recent_proj_list[:max_recent]
    # Substitute empty module device with global default device
    try:
        from modules.base import DEFAULT_DEVICE
        default_dev = (getattr(pcfg, 'default_device', None) or '').strip() or DEFAULT_DEVICE
        for param_key in ('textdetector_params', 'ocr_params', 'translator_params', 'inpainter_params'):
            d = getattr(pcfg.module, param_key, None)
            if not d:
                continue
            for mod_name, mod_params in d.items():
                if not isinstance(mod_params, dict) or 'device' not in mod_params:
                    continue
                dev = mod_params.get('device')
                if isinstance(dev, dict) and (not (dev.get('value') or '').strip()):
                    dev['value'] = default_dev
    except Exception:
        pass

    # Section 8: apply HuggingFace token from config so gated models and Xet use it
    try:
        token = (getattr(pcfg, 'huggingface_token', None) or '').strip()
        if token:
            from utils.model_manager import get_model_manager
            get_model_manager().set_hf_token(token)
    except Exception:
        pass

    p = (pcfg.text_styles_path or '').strip()
    if not p:
        p = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')
        pcfg.text_styles_path = p
    p = osp.normpath(osp.abspath(p))
    pcfg.text_styles_path = p
    dp = osp.normpath(osp.abspath(osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')))
    if not osp.exists(p):
        if p != dp and osp.exists(dp):
            p = dp
            pcfg.text_styles_path = p
            LOGGER.warning(f'Text style path missing; using default at {dp}.')
        else:
            try:
                os.makedirs(osp.dirname(dp), exist_ok=True)
            except Exception:
                pass
            with open(dp, 'w', encoding='utf8') as f:
                f.write(json.dumps([],  ensure_ascii=False))
            LOGGER.info(f'New text style file created at {dp}.')
            p = dp
            pcfg.text_styles_path = p
    load_textstyle_from(p)

    # Create config.json on first run so ZIP users get recommended defaults from config.example.json
    if not osp.exists(shared.CONFIG_PATH):
        save_config()


def json_dump_program_config(obj, **kwargs):
    def _default(o):
        if isinstance(o, (np.ndarray, np.ScalarType)):
            return serialize_np(o)
        elif isinstance(o, ModuleConfig):
            return o.get_saving_params()
        elif type(o).__name__ == "ProgramConfig":
            # Section 9: canonical key order for clean diffs
            ordered = {}
            for k in CONFIG_KEY_ORDER:
                if hasattr(o, k):
                    ordered[k] = getattr(o, k)
            for k in o.__dict__:
                if k not in ordered:
                    ordered[k] = getattr(o, k)
            return ordered
        return o.__dict__
    return json.dumps(obj, default=lambda o: _default(o), ensure_ascii=False, **kwargs)


def save_config():
    """Save program config to user config file (config/config.json). Never writes to config.example.json."""
    global pcfg
    if osp.basename(shared.CONFIG_PATH) == 'config.example.json':
        LOGGER.warning('Refusing to save to config.example.json; user config is config.json (gitignored).')
        return False
    try:
        tmp_save_tgt = shared.CONFIG_PATH + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            f.write(json_dump_program_config(pcfg))
    except Exception as e:
        LOGGER.error(f'Failed save config to {tmp_save_tgt}: {e}')
        LOGGER.error(traceback.format_exc())
        return False

    os.replace(tmp_save_tgt, shared.CONFIG_PATH)
    LOGGER.info('Config saved')
    return True

def save_text_styles(raise_exception = False):
    global pcfg, text_styles
    try:
        style_dir = osp.dirname(pcfg.text_styles_path)
        if not osp.exists(style_dir):
            os.makedirs(style_dir)
        tmp_save_tgt = pcfg.text_styles_path + '.tmp'
        with open(tmp_save_tgt, 'w', encoding='utf8') as f:
            f.write(json_dump_nested_obj(text_styles))

    except Exception as e:
        LOGGER.error(f'Failed save text style to {tmp_save_tgt}: {e}')
        LOGGER.error(traceback.format_exc())
        if raise_exception:
            raise e
        return False

    os.replace(tmp_save_tgt, pcfg.text_styles_path)
    LOGGER.info('Text style saved')
    return True