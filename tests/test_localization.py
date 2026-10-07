import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules import localization


class TestLocalization(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.patch_root = patch.object(localization, 'localization_root', str(self.root))
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)
        self.patch_translation = patch.object(localization, 'current_translation', {})
        self.patch_translation.start()
        self.addCleanup(self.patch_translation.stop)
        (self.root / 'zh.json').write_text(json.dumps({
            'Ready: {count} entries.': '已就绪：{count} 条。',
            'prefix': '前缀匹配',
            'None': '无',
            'HTML': '</script><b>文字</b>',
        }), encoding='utf-8')
        localization.load_localization('zh')

    def test_parameters_and_english_fallback(self):
        self.assertEqual(localization.translate('Ready: {count} entries.', {'count': 120034}), '已就绪：120034 条。')
        self.assertEqual(localization.translate('Ready: {count} entries.', {'count': 0}), '已就绪：0 条。')
        self.assertEqual(localization.translate('Unknown: {name}', {'name': '$& {other}'}), 'Unknown: $& {other}')
        self.assertEqual(localization.translate('Ready: {count} entries.'), '已就绪：{count} 条。')
        self.assertEqual(localization.translate('{"name":"cat"}'), '{"name":"cat"}')

    def test_choice_labels_do_not_change_values(self):
        self.assertEqual(localization.translate_choices(['prefix', 'None', 'cat.safetensors']), [
            ('前缀匹配', 'prefix'), ('无', 'None'), ('cat.safetensors', 'cat.safetensors'),
        ])
        self.assertEqual(localization.translate_choices([('prefix', 'raw'), ['None', 0]]), [
            ('前缀匹配', 'raw'), ('无', 0),
        ])

    def test_missing_and_invalid_languages_do_not_retain_previous_dictionary(self):
        for content in [None, '{', '[]', '{"prefix":1}']:
            with self.subTest(content=content):
                localization.load_localization('zh')
                if content is not None:
                    (self.root / 'bad.json').write_text(content, encoding='utf-8')
                with patch('builtins.print'):
                    localization.load_localization('missing' if content is None else 'bad')
                self.assertEqual(localization.translate('prefix'), 'prefix')

    def test_script_payload_is_valid_json_without_closing_script_tags(self):
        script = localization.localization_js('zh')
        self.assertNotIn('</script>', script)
        self.assertEqual(json.loads(script.removeprefix('window.localization = '))['HTML'], '</script><b>文字</b>')


if __name__ == '__main__':
    unittest.main()
