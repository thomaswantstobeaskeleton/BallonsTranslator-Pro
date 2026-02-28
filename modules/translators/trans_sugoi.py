from .base import *
from .exceptions import MissingTranslatorParams
import ctranslate2, sentencepiece as spm
import os

import utils.shared as shared
SUGOIMODEL_TRANSLATOR_DIRPATH = os.path.join(shared.PROGRAM_PATH, 'data', 'models', 'sugoi_translator')
SUGOIMODEL_TOKENIZATOR_PATH = os.path.join(SUGOIMODEL_TRANSLATOR_DIRPATH, "spm.ja.nopretok.model")
SUGOIMODEL_BIN = os.path.join(SUGOIMODEL_TRANSLATOR_DIRPATH, "model.bin")


@register_translator('Sugoi')
class SugoiTranslator(BaseTranslator):

    concate_text = False
    params: Dict = {
        'device': DEVICE_SELECTOR()
    }

    def _setup_translator(self):
        self.translator = None
        self.tokenizator = None
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        if not os.path.isfile(SUGOIMODEL_BIN):
            return  # Defer error to first translate so compatibility check can pass
        self._load_model()

    def _load_model(self):
        if not os.path.isfile(SUGOIMODEL_BIN):
            raise MissingTranslatorParams(
                f"Sugoi model not found. Please download the Sugoi CTranslate2 model and place it in:\n{os.path.normpath(SUGOIMODEL_TRANSLATOR_DIRPATH)}\n"
                "The folder must contain model.bin and spm.ja.nopretok.model. See project docs for download links."
            )
        device = self.params.get('device')
        if isinstance(device, dict) and 'value' in device:
            device = device['value']
        self.translator = ctranslate2.Translator(SUGOIMODEL_TRANSLATOR_DIRPATH, device=device or 'cpu')
        self.tokenizator = spm.SentencePieceProcessor(model_file=SUGOIMODEL_TOKENIZATOR_PATH)

    def _translate(self, src_list: List[str]) -> List[str]:
        if self.translator is None:
            self._load_model()

        text = [i.replace(".", "@").replace("．", "@") for i in src_list]
        tokenized_text = self.tokenizator.encode(text, out_type=str, enable_sampling=True, alpha=0.1, nbest_size=-1)
        tokenized_translated = self.translator.translate_batch(tokenized_text)
        text_translated = [''.join(text[0]["tokens"]).replace('▁', ' ').replace("@", ".") for text in tokenized_translated]
        
        return text_translated

    def updateParam(self, param_key: str, param_content):
        super().updateParam(param_key, param_content)
        if param_key == 'device' and self.translator is not None and os.path.isfile(SUGOIMODEL_BIN):
            device = self.params.get('device')
            if isinstance(device, dict) and 'value' in device:
                device = device['value']
            if hasattr(self, 'translator'):
                delattr(self, 'translator')
            self.translator = ctranslate2.Translator(SUGOIMODEL_TRANSLATOR_DIRPATH, device=device or 'cpu')

    @property
    def supported_tgt_list(self) -> List[str]:
        return ['English']

    @property
    def supported_src_list(self) -> List[str]:
        return ['日本語']