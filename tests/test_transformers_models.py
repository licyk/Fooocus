import tempfile
import unittest

import torch
from transformers import BertConfig, BertModel

from extras.BLIP.models.blip import init_tokenizer
from extras.BLIP.models.med import BertLMHeadModel
from extras.GroundingDINO.bert import wrap_bert_model


class TestTransformersModels(unittest.TestCase):
    def test_bundled_blip_tokenizer_encoder_special_token(self):
        tokenizer = init_tokenizer()
        self.assertNotEqual(tokenizer.enc_token_id, tokenizer.unk_token_id)
        self.assertEqual(tokenizer.convert_ids_to_tokens(tokenizer.enc_token_id), '[ENC]')
        self.assertEqual(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), '[DEC]')

    def setUp(self):
        self.config = BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32, encoder_width=16,
            add_cross_attention=True, is_decoder=True,
            bos_token_id=1, eos_token_id=2, pad_token_id=0,
        )
        self.ids = torch.tensor([[1, 4, 5, 3]])
        self.encoder = torch.randn(1, 3, 16)

    def test_blip_cached_logits_match_full_sequence(self):
        model = BertLMHeadModel(self.config).eval()
        with torch.no_grad():
            full = model(self.ids, encoder_hidden_states=self.encoder).logits[:, -1]
            prefix = model(self.ids[:, :-1], encoder_hidden_states=self.encoder)
            cached = model(
                self.ids[:, -1:], attention_mask=torch.ones_like(self.ids),
                encoder_hidden_states=self.encoder, past_key_values=prefix.past_key_values,
            ).logits[:, -1]
        torch.testing.assert_close(full, cached)

    def test_blip_beam_search_expands_encoder_and_reorders_cache(self):
        model = BertLMHeadModel(self.config).eval()
        result = model.generate(
            self.ids, encoder_hidden_states=self.encoder, num_beams=3,
            max_new_tokens=3, min_new_tokens=3,
        )
        self.assertEqual(result.shape, (1, 7))

    def test_blip_checkpoint_preserves_parameters_and_tied_embeddings(self):
        model = BertLMHeadModel(self.config).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = BertLMHeadModel.from_pretrained(directory).eval()
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, restored.state_dict()[name])
        self.assertEqual(
            restored.get_input_embeddings().weight.data_ptr(),
            restored.get_output_embeddings().weight.data_ptr(),
        )

    def test_groundingdino_preserves_weights_and_accepts_token_masks(self):
        config = BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32,
        )
        model = BertModel(config).eval()
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.no_grad():
            expected = model(self.ids, attention_mask=torch.ones_like(self.ids)).last_hidden_state
            model = wrap_bert_model(model)
            actual = model(self.ids, attention_mask=torch.ones(1, 4, 4, dtype=torch.bool)).last_hidden_state
            isolated = model(self.ids, attention_mask=torch.eye(4, dtype=torch.bool)[None]).last_hidden_state
        torch.testing.assert_close(actual, expected)
        self.assertFalse(torch.allclose(isolated, expected))
        self.assertTrue(torch.isfinite(isolated).all())
        self.assertEqual(set(model.state_dict()), set(weights))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[name])
