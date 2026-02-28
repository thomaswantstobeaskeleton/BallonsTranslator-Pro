# Stub for Ocean-OCR processor_ocean when tos (Volcengine TOS) SDK is not installed.
# Only used for cloud paths; BallonsTranslator uses local paths only.


class TosClientV2:
    def __init__(self, ak, sk, endpoint, region):
        pass

    def get_object(self, bucket_name, path):
        class Stream:
            def read(self):
                return None

        return Stream()
