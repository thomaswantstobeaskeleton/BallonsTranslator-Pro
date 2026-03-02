import json, os, traceback
import os.path as osp
import copy
from typing import Callable

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
    finish_code: int = 15
    run_preset_name: str = 'Full'

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
    shortcuts: Dict = field(default_factory=dict)
    auto_region_merge_after_run: str = 'never'  # 'never' | 'all_pages' | 'current_page'
    region_merge_settings: Dict = field(default_factory=dict)  # Region merge tool dialog (persisted)
    context_menu: Dict = field(default_factory=dict)  # Canvas right-click: action key -> visible (default True)

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
    'format_apply': True, 'format_layout': True, 'format_angle': True, 'format_squeeze': True,
    'run_detect_region': True, 'run_detect_page': True, 'run_translate': True, 'run_ocr': True,
    'run_ocr_translate': True, 'run_ocr_translate_inpaint': True, 'run_inpaint': True,
}

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

    if osp.exists(shared.CONFIG_PATH):
        try:
            config = ProgramConfig.load(shared.CONFIG_PATH)
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.warning("Failed to load config file, using default config")
            config = ProgramConfig()
    else:
        LOGGER.info(f'{shared.CONFIG_PATH} does not exist, new config file will be created.')
        config = ProgramConfig()
    
    global pcfg
    pcfg.merge(config)
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

    p = pcfg.text_styles_path
    if not osp.exists(pcfg.text_styles_path):
        dp = osp.join(shared.DEFAULT_TEXTSTYLE_DIR, 'default.json')
        if p != dp and osp.exists(dp):
            p = dp
            LOGGER.warning(f'Text style {p} does not exist, use the default from {dp}.')
        else:
            with open(dp, 'w', encoding='utf8') as f:
                f.write(json.dumps([],  ensure_ascii=False))
            LOGGER.info(f'New text style file created at {dp}.')
    load_textstyle_from(p)


def json_dump_program_config(obj, **kwargs):
    def _default(obj):
        if isinstance(obj, (np.ndarray, np.ScalarType)):
            return serialize_np(obj)
        elif isinstance(obj, ModuleConfig):
            return obj.get_saving_params()
        return obj.__dict__
    return json.dumps(obj, default=lambda o: _default(o), ensure_ascii=False, **kwargs)


def save_config():
    global pcfg
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