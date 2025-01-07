import unittest
import tensorflow as tf
import numpy as np

from src.model import ImageCaptionModel


class TestImageCaptionModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.max_length = 20
        self.model = ImageCaptionModel(self.vocab_size, self.max_length)
        self.test_image = np.random.random((1, 224, 224, 3))
        self.test_sequence = np.random.randint(0, self.vocab_size, (1, self.max_length))

    def test_encoder_architecture(self):
        encoded = self.model.encoder.predict(self.test_image)
        self.assertEqual(encoded.shape[-1], self.model.embedding_dim)

        frozen_layers = sum(1 for layer in self.model.encoder.layers if not layer.trainable)
        self.assertGreater(frozen_layers, 0)

    def test_decoder_architecture(self):
        encoded = self.model.encoder.predict(self.test_image)

        output = self.model.decoder.predict([encoded, self.test_sequence])

        self.assertEqual(output.shape[-1], self.vocab_size)
        self.assertTrue(np.allclose(output.sum(axis=1), 1.0))

    def test_model_compilation(self):
        compiled_model = self.model.compile_model()

        self.assertIsNotNone(compiled_model.optimizer)
        self.assertEqual(compiled_model.loss, 'categorical_crossentropy')
        self.assertTrue(len(compiled_model.metrics) > 0)


if __name__ == '__main__':
    unittest.main()