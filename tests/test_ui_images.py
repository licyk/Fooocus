import unittest

import numpy as np

from modules.ui_images import editor_to_inpaint


class TestEditorImages(unittest.TestCase):
    def test_empty_editor(self):
        self.assertIsNone(editor_to_inpaint(None))
        self.assertIsNone(editor_to_inpaint({'background': None, 'layers': [], 'composite': None}))

    def test_background_is_preserved_without_a_mask(self):
        image = np.full((3, 4, 3), 60, dtype=np.uint8)
        result = editor_to_inpaint({'background': image, 'layers': []})
        np.testing.assert_array_equal(result['image'], image)
        np.testing.assert_array_equal(result['mask'], np.zeros_like(image))

    def test_mask_uses_layer_alpha_and_keeps_original_pixels(self):
        image = np.full((3, 4, 3), 60, dtype=np.uint8)
        layer = np.zeros((3, 4, 4), dtype=np.uint8)
        layer[1, 2] = [0, 0, 0, 255]
        result = editor_to_inpaint({'background': image, 'layers': [layer], 'composite': layer})
        np.testing.assert_array_equal(result['image'], image)
        self.assertEqual(result['mask'][1, 2].tolist(), [255, 255, 255])
        self.assertEqual(result['mask'][0, 0].tolist(), [0, 0, 0])

    def test_multiple_layers_combine_coverage(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        first = np.zeros((2, 2, 4), dtype=np.uint8)
        second = first.copy()
        first[0, 0, 3] = 128
        second[0, 0, 3] = 128
        second[1, 1, 3] = 255
        result = editor_to_inpaint({'background': image, 'layers': [first, second]})
        self.assertEqual(result['mask'][0, 0, 0], 192)
        self.assertEqual(result['mask'][1, 1, 0], 255)
        self.assertEqual(result['mask'][0, 1, 0], 0)

    def test_transparent_background_is_composited_on_white(self):
        image = np.zeros((2, 2, 4), dtype=np.uint8)
        image[0, 0] = [10, 20, 30, 255]
        result = editor_to_inpaint({'background': image, 'layers': []})
        self.assertEqual(result['image'][0, 0].tolist(), [10, 20, 30])
        self.assertEqual(result['image'][1, 1].tolist(), [255, 255, 255])

    def test_composite_is_used_when_background_is_missing(self):
        image = np.full((2, 2, 3), 120, dtype=np.uint8)
        result = editor_to_inpaint({'background': None, 'layers': [], 'composite': image})
        np.testing.assert_array_equal(result['image'], image)
