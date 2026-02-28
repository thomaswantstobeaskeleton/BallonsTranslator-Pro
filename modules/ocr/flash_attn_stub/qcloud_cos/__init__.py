# Stub for Ocean-OCR processor_ocean when cos-python-sdk-v5 is not installed.
# Only used for cloud paths; BallonsTranslator uses local paths only.


class CosConfig:
    def __init__(self, **kwargs):
        pass


class CosS3Client:
    def __init__(self, config):
        self._config = config

    def get_object(self, Bucket=None, Key=None):
        class Body:
            def get_raw_stream(self):
                return self

            def read(self):
                return b""

        return {"Body": Body()}
