import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules import localization
from modules.html import make_progress_html


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


class TestGenerationProgress(unittest.TestCase):
    def setUp(self):
        dictionary = json.loads((Path(__file__).resolve().parents[1] / 'language' / 'zh.json').read_text(encoding='utf-8'))
        translation = patch.object(localization, 'current_translation', dictionary)
        translation.start()
        self.addCleanup(translation.stop)

    def test_progress_preserves_image_numbers_steps_and_dimensions(self):
        cases = [
            ('Sampling step {step}/{steps}, image {index}/{total} ...',
             {'step': 1, 'steps': 40, 'index': 1, 'total': 30},
             '正在生成图片 1/30，采样步数 1/40…'),
            ('Sampling step {step}/{steps}, image {index}/{total} ...',
             {'step': 40, 'steps': 40, 'index': 30, 'total': 30},
             '正在生成图片 30/30，采样步数 40/40…'),
            ('Saving image {index}/{total} to system ...',
             {'index': 30, 'total': 30}, '正在保存图片 30/30…'),
            ('Preparing enhancement {index}/{total} ...',
             {'index': 7, 'total': 90}, '正在准备增强任务 7/90…'),
            ('Encoding positive #{index} ...', {'index': 2}, '正在编码第 2 组正面提示词…'),
            ('Encoding negative #{index} ...', {'index': 2}, '正在编码第 2 组负面提示词…'),
            ('Upscaling image from ({width}, {height}) ...',
             {'width': 1152, 'height': 896}, '正在放大图片，原始尺寸 1152×896…'),
            ('Progress {percent}%', {'percent': 0}, '进度 0%'),
        ]
        for source, params, expected in cases:
            with self.subTest(source=source, params=params):
                html = make_progress_html(42, source, params)
                self.assertIn(f'<span>{expected}</span>', html)
                self.assertIn('<progress value="42" max="100">', html)
                self.assertNotIn('{', html)

    def test_initial_wait_and_generation_phases_are_translated(self):
        cases = {
            'Waiting for task to start ...': '等待任务开始 ...',
            'Checking for NSFW content ...': '正在检查 NSFW 内容…',
            'VAE Refiner encoding ...': '正在进行精修模型的 VAE 编码…',
            'Preparing enhance prompts ...': '正在准备增强提示词…',
            'Processing enhance ...': '正在增强图片…',
            'Saving image to system ...': '正在保存图片…',
            'Loading ...': '正在加载…',
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                self.assertIn(f'<span>{expected}</span>', make_progress_html(1, source))

    def test_english_fallback_and_status_text_are_safe_in_html(self):
        with patch.object(localization, 'current_translation', {}):
            self.assertIn('Sampling step 3/40, image 2/30 ...', make_progress_html(
                10, 'Sampling step {step}/{steps}, image {index}/{total} ...',
                {'step': 3, 'steps': 40, 'index': 2, 'total': 30},
            ))
            html = make_progress_html(10, 'Loading {name}', {'name': '<model> & weights'})
            self.assertIn('<span>Loading &lt;model&gt; &amp; weights</span>', html)
            self.assertNotIn('<model>', html)


if __name__ == '__main__':
    unittest.main()
