import unittest
import numpy as np
import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train import Trainer
from unittest.mock import patch


class TestTrainer(unittest.TestCase):
    def setUp(self):
        self.data_dir = "data"
        self.captions_file = "../data/captions.txt"
        self.trainer = Trainer(self.data_dir, self.captions_file, batch_size=2, epochs=1, initial_learning_rate=0.001)

    @patch('src.data_loader.DataLoader.load_image')
    def test_data_generator(self, mock_load_image):
        batch_size = 2
        mock_load_image.return_value = np.random.random((batch_size, 224, 224, 3))

        generator = self.trainer.create_data_generator(self.trainer.captions_dict)
        inputs, targets = next(generator)

        self.assertEqual(inputs[0].shape[1:], (224, 224, 3))
        self.assertEqual(inputs[1].shape[1:], (self.trainer.max_length - 1,))
        self.assertEqual(targets.shape[1:], (self.trainer.max_length - 1,))

    @patch('src.data_loader.DataLoader.load_image')
    def test_train_step(self, mock_load_image):
        mock_load_image.return_value = np.random.random((224, 224, 3))

        images = tf.random.uniform((2, 224, 224, 3))
        captions = tf.random.uniform((2, self.trainer.max_length), minval=0, maxval=100, dtype=tf.int32)

        trainer = self.trainer
        image_features = trainer.model.encoder(images)

        with tf.GradientTape() as tape:
            # Convert inputs to tensors with consistent types
            decoder_inputs = [
                tf.convert_to_tensor(image_features, dtype=tf.float32),
                tf.convert_to_tensor(captions, dtype=tf.int32)
            ]
            predictions = trainer.model.decoder(decoder_inputs)
            loss = trainer.model.decoder.compiled_loss(
                captions,
                predictions,
                regularization_losses=trainer.model.decoder.losses
            )

        gradients = tape.gradient(loss, trainer.model.decoder.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        trainer.model.decoder.optimizer.apply_gradients(
            zip(gradients, trainer.model.decoder.trainable_variables)
        )

        self.assertTrue(np.isfinite(loss.numpy()))


if __name__ == "__main__":
    unittest.main()