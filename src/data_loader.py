import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, image_dir, captions_file, max_length, vocab_size=None, tokenizer=None):
        self.image_dir = image_dir
        self.captions_file = captions_file
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

        self.captions_df = pd.read_csv(captions_file)
        self.image_filenames = self.captions_df['image'].tolist()
        self.captions = self.captions_df['caption'].tolist()

        self.captions_dict = {}
        for img, cap in zip(self.image_filenames, self.captions):
            if img not in self.captions_dict:
                self.captions_dict[img] = []
            self.captions_dict[img].append(cap)

        if not self.tokenizer:
            self.tokenizer = self._create_tokenizer(self.captions)

        if not self.vocab_size:
            self.vocab_size = len(self.tokenizer.word_index) + 1

        print(f"Vocabulary size: {self.vocab_size}")

    def _create_tokenizer(self, captions):
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(captions)
        return tokenizer

    def _load_image(self, filename):
        try:
            image_path = os.path.join(self.image_dir, filename)

            print(f"Attempting to load image from: {image_path}")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                raise ValueError(f"Unsupported image format: {filename}")


            image = load_img(image_path, target_size=(256, 256))
            image = img_to_array(image)
            image = preprocess_input(image)

            return image

        except Exception as e:
            print(f"Error loading image {filename}:")
            print(f"Image directory: {self.image_dir}")
            print(f"Full path: {os.path.abspath(image_path)}")
            print(f"Error details: {str(e)}")
            raise

    def _process_caption(self, caption):

        sequence = self.tokenizer.texts_to_sequences([caption])[0]
        padded_sequence = pad_sequences([sequence], maxlen=self.max_length, padding='post')[0]
        return padded_sequence

    def generate_data(self):

        images = []
        captions_input = []
        captions_output = []


        for image_filename, caption in zip(self.image_filenames, self.captions):

            image = self._load_image(image_filename)
            images.append(image)

            caption_seq = self._process_caption(caption)
            captions_input.append(caption_seq[:-1])
            captions_output.append(caption_seq[1:])

        images = np.array(images)
        captions_input = np.array(captions_input)
        captions_output = np.array(captions_output)

        return images, captions_input, captions_output

    def train_test_split(self, test_size=0.2, random_state=42):

        images, captions_input, captions_output = self.generate_data()

        train_images, test_images, train_input, test_input, train_output, test_output = train_test_split(
            images, captions_input, captions_output, test_size=test_size, random_state=random_state
        )

        return (train_images, train_input, train_output), (test_images, test_input, test_output)
