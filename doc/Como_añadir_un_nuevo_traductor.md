[简体中文](../doc/加别的翻译器.md) | [English](../doc/how_to_add_new_translator.md) | [pt-BR](../doc/Como_add_um_novo_tradutor.md)) | [Русский](../doc/add_translator_ru.md) | ESPAÑOL

---

## Cómo añadir un nuevo traductor a BallonsTranslator

Si sabe utilizar la API del traductor o el modelo de traducción deseado en Python, siga los pasos que se indican a continuación para integrarlo con BallonsTranslator.

### Implementación de la clase traductor

Si sabes cómo llamar a la API del traductor de destino o del modelo de traducción en Python, implementa una clase en `ballontranslator/dl/translators.__init__.py` como se indica a continuación para utilizarla en la aplicación. El siguiente ejemplo, DummyTranslator, está comentado en `ballontranslator/dl/translator/__init__.py` y puede descomentarse para probarlo en el programa.

1. **Crear una nueva clase en `ballontranslator/dl/translators/__init__.py`:**

```python
# "dummy translator" es el nombre que aparece en la aplicación
@register_translator('dummy translator')
class DummyTranslator(BaseTranslator):

    concate_text = True

    # parámetros mostrados en el panel de configuración.
    # las claves son los nombres de los parámetros, si el tipo del valor es str, será un editor de texto (clave requerida)
    # si el tipo del valor es dict, es necesario especificar el 'tipo' del parámetro,
    # el siguiente 'device' es un selector, las opciones son cpu y cuda, por defecto es cpu
    params: Dict = {
        'api_key': '', 
        'device': {
            'type': 'selector',
            'options': ['cpu', 'cuda'],
            'value': 'cpu'
        }
    }

    def _setup_translator(self):
        '''
        configúrelo aquí.
        las claves lang_map son las opciones de idioma que se muestran en la aplicación,
        asigne las correspondientes claves de idioma aceptadas por la API a los idiomas soportados.
        Aquí sólo se asignan los idiomas soportados por el traductor, este traductor sólo soporta japonés e inglés.
        Para obtener una lista completa de idiomas, consulte LANGMAP_GLOBAL en translator.__init__.
        '''
        self.lang_map['日本語'] = 'ja'
        self.lang_map['English'] = 'en'
        
    def _translate(self, src_list: List[str]) -> List[str]:
        '''
        hacer la traducción aquí.
        Este traductor no hace más que devolver el texto original.
        '''
        source = self.lang_map[self.lang_source]
        target = self.lang_map[self.lang_target]
        
        translation = text
        return translation

    def updateParam(self, param_key: str, param_content):
        '''
        sólo es necesario si algún estado necesita ser actualizado inmediatamente después de que el usuario cambie los parámetros del traductor,
        por ejemplo, si este traductor es un modelo pytorch, puedes convertirlo a cpu/gpu aquí.
        '''
        super().updateParam(param_key, param_content)
        if (param_key == 'device'):
            # obtener el estado actual de los parámetros
            # self.model.to(self.params['device']['value'])
            pass

    @property
    def supported_tgt_list(self) -> List[str]:
        '''
        sólo es necesario si el soporte lingüístico del traductor es asimétrico,
        por ejemplo, este traductor sólo admite inglés -> japonés, no japonés -> inglés.
        '''
        return ['English']

    @property
    def supported_src_list(self) -> List[str]:
        '''
        sólo es necesario si el soporte lingüístico del traductor es asimétrico.
        '''
        return ['日本語']
```

- Decora la clase con `@register_translator` y proporciona el nombre del traductor que se mostrará en la interfaz. En el ejemplo, el nombre pasado al decorador es `'dummy translator'`, tenga cuidado de no renombrarlo con un traductor existente.
- La clase debe heredar de `BaseTranslator`.

2. **Establece el atributo `concate_text`:**

```python
@register_translator('dummy translator')
class DummyTranslator(BaseTranslator):  
    concate_text = True  # Si el traductor sólo acepta cadenas concatenadas
    concate_text = False # Si el traductor acepta listas de cadenas o plantillas offline
```

- Indica si el traductor sólo acepta texto concatenado (varias frases en una sola cadena) o una lista de cadenas.
- Si se trata de un modelo offline o de una API que acepta listas de cadenas, establézcalo en `False`.

3. **Establezca los parámetros (opcional):**

```python
params: Dict = {
    'api_key': '',  # Editor de texto para la clave API
    'device': {    # Selector para CPU o CUDA
        'type': 'selector',
        'options': ['cpu', 'cuda'],
        'value': 'cpu'
    }
}
```

- Crea un diccionario `params` si el traductor necesita parámetros configurables por el usuario. Si no, déjelo en blanco o asígnele `None`.
- Las claves del diccionario son los nombres de los parámetros que se muestran en la interfaz. Si el tipo de valor correspondiente es str, se mostrará en la aplicación como un editor de texto; en el ejemplo anterior, api_key será un editor de texto con un valor por defecto vacío.
- Los valores pueden ser cadenas (para editores de texto) o diccionarios (en cuyo caso deben describirse mediante 'type', como en el ejemplo anterior. El parámetro 'device' se mostrará como un selector en la aplicación, las opciones válidas son 'cpu' y 'cuda).

<p align="center">
<img src="./src/new_translator.png">
</p>
<p align="center">
parámetros mostrados en el panel de configuración de la aplicación.
</p>  

4. **Implementa el método `_setup_translator`:**

```python
def _setup_translator(self):
    '''
    configúrelo aquí.
    las claves lang_map son las opciones de idioma que se muestran en la aplicación,
    asigne las correspondientes claves de idioma aceptadas por la API a los idiomas soportados.
    Aquí sólo se asignan los idiomas soportados por el traductor, este traductor sólo soporta japonés e inglés.
    Para obtener una lista completa de idiomas, consulte LANGMAP_GLOBAL en translator.__init__.
    '''
    self.lang_map['日本語'] = 'ja'
    self.lang_map['English'] = 'en'
```

- Configurar el traductor (inicialización del modelo, autenticación de la API, etc.).
- Asignar los idiomas mostrados en la aplicación a los códigos de idioma aceptados por la API.
- Consulta `LANGMAP_GLOBAL` en `translator.__init__` para ver la lista completa de idiomas.

5. **Implementa el método `_translate`:**

```python
def _translate(self, src_list: List[str]) -> List[str]:
    '''
    hacer la traducción aquí.
    Este traductor no hace más que devolver el texto original.
    '''
    source = self.lang_map[self.lang_source]
    target = self.lang_map[self.lang_target]
    
    translation = text
    return translation
```

- Recibe una lista de cadenas (`src_list`) para traducir.
- Si `concate_text` es `True`, las cadenas se concatenarán antes de pasarlas al traductor.
- Realiza la traducción utilizando la API o el modelo.
- Devuelve una lista de cadenas traducidas.

### Métodos opcionales

- **`updateParam(self, param_key: str, param_content)`:**
    - Impleméntelo si necesita actualizar el estado del traductor inmediatamente después de que el usuario cambie los parámetros.

- **`supported_tgt_list(self) -> List[str]`:**
    - Implementar si el soporte lingüístico del traductor es asimétrico (por ejemplo, sólo traduce del inglés al japonés).

- **`supported_src_list(self) -> List[str]`:**
    - Aplicar si el soporte lingüístico del traductor es asimétrico.

### Pruebas

Después de implementar el traductor, pruébalo seleccionándolo en la aplicación (Config → Translator) y ejecutando una traducción.