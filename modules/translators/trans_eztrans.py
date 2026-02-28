import utils.shared as shared

if shared.ON_WINDOWS:
    from .base import *
    import os
    from typing import Literal
    from msl.loadlib import Client64


    class MyClient(Client64):
        def __init__(self, engine_path, engine_type: Literal['J2K', 'K2J'], dat_path):
            super(MyClient, self).__init__(module32=str(os.path.dirname(os.path.realpath(__file__))) + '/module_eztrans32.py',
                                        engine_path=engine_path,
                                        engine_type=engine_type,
                                        dat_path=dat_path)

        def translate(self, src_text: Union[str, list]):
            return self.request32('translate', src_text)


    def fullwidth_to_halfwidth(text):
        mapping = {i: i - 0xFEE0 for i in range(0xFF01, 0xFF5F)}
        mapping[0x3000] = 0x0020  # 전각 공백 → 반각 공백
        return text.translate(mapping)

    @register_translator('ezTrans')
    class ezTransTranslator(BaseTranslator):
        concate_text = True

        params: Dict = {
            'path_dat': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\Dat",
            'path_j2k(J2KEngine.dll)': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\J2KEngine.dll",
            'path_k2j(ehnd-kor.dll, Optional)': r"C:\Program Files (x86)\ChangShinSoft\ezTrans XP\ehnd-kor.dll"
        }

        def _get_path(self, key: str) -> str:
            p = self.params.get(key)
            if isinstance(p, dict) and 'value' in p:
                return (p['value'] or '').strip()
            return (p or '').strip() if isinstance(p, str) else ''

        def _setup_translator(self):
            self.textblk_break = '\n'
            self.lang_map['日本語'] = 'j'
            self.lang_map['한국어'] = 'k'

            self.j2k_engine, self.k2j_engine = (None, None)

            path_j2k = self._get_path('path_j2k(J2KEngine.dll)')
            path_k2j = self._get_path('path_k2j(ehnd-kor.dll, Optional)')
            path_dat = self._get_path('path_dat')
            if path_j2k and os.path.exists(path_j2k):
                self.j2k_engine = MyClient(path_j2k, "J2K", path_dat)
            if path_k2j and os.path.exists(path_k2j):
                self.k2j_engine = MyClient(path_k2j, "K2J", path_dat)

        def _translate(self, src_list: List[str]) -> List[str]:
            source = self.lang_map[self.lang_source]
            target = self.lang_map[self.lang_target]

            if source != target:
                engine: MyClient = getattr(self, f"{source}2{target}_engine")
                return engine.translate(src_list) if source != "k" else fullwidth_to_halfwidth(engine.translate(src_list))
            else:
                return src_list

        def updateParam(self, param_key: str, param_content):
            super().updateParam(param_key, param_content)

            path_j2k = self._get_path('path_j2k(J2KEngine.dll)')
            path_k2j = self._get_path('path_k2j(ehnd-kor.dll, Optional)')
            path_dat = self._get_path('path_dat')
            if not self.j2k_engine and path_j2k and os.path.exists(path_j2k):
                self.j2k_engine = MyClient(path_j2k, "J2K", path_dat)
            if not self.k2j_engine and path_k2j and os.path.exists(path_k2j):
                self.k2j_engine = MyClient(path_k2j, "K2J", path_dat)

        @property
        def supported_tgt_list(self) -> List[str]:
            return ['한국어', '日本語'] if self.j2k_engine else ['한국어']

        @property
        def supported_src_list(self) -> List[str]:
            return ['한국어', '日本語'] if self.k2j_engine else ['日本語']
