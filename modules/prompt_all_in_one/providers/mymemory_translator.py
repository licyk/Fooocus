# Ported from Physton/sd-webui-prompt-all-in-one (MIT).

from .base import BaseTranslator, get_lang, requests


class MyMemoryTranslator(BaseTranslator):
    def __init__(self):
        super().__init__("mymemory")

    def translate(self, text):
        if not text:
            return ""
        url = "https://api.mymemory.translated.net/get"
        api_key = self.api_config.get("api_key", "")
        params = {
            "q": text,
            "langpair": f"{self.from_lang}|{self.to_lang}",
        }
        if api_key:
            params["key"] = api_key

        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(get_lang("request_error", {"0": "myMemory"}))
        if not response.text:
            raise Exception(get_lang("response_is_empty", {"0": "myMemory"}))
        result = response.json()
        if "responseStatus" not in result:
            raise Exception(get_lang("no_response_from", {"0": "myMemory"}))
        if result["responseStatus"] != 200:
            raise Exception(result["responseDetails"])
        if "responseData" not in result:
            raise Exception(get_lang("no_response_from", {"0": "myMemory"}))
        if "translatedText" not in result["responseData"]:
            raise Exception(get_lang("no_response_from", {"0": "myMemory"}))
        return result["responseData"]["translatedText"]
