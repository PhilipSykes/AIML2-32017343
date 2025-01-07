import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


class CaptionGenerator:
    def __init__(self, model_path, tokenizer, max_length=40):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab_size = len(self.tokenizer.word_index) + 1


        base_model = ResNet50(weights="imagenet", include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        self.encoder = Model(inputs=base_model.input, outputs=x)

        self.decoder = load_model(model_path)

    def preprocess_image(self, image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        return image

    def generate_caption(self, image_path, beam_width=5, alpha=0.7):
        image = self.preprocess_image(image_path)
        image_features = self.encoder.predict(image, verbose=0)

        beams = [([], 0.0)]
        completed_beams = []

        for _ in range(self.max_length - 1):
            candidates = []

            for seq, score in beams:
                if seq and seq[-1] == 1:
                    completed_beams.append((seq, score))
                    continue

                input_seq = seq + [1] if not seq else seq
                input_seq = tf.keras.preprocessing.sequence.pad_sequences(
                    [input_seq], maxlen=self.max_length - 1, padding='post'
                )

                predictions = self.decoder.predict(
                    [image_features, input_seq], verbose=0
                )[0, len(seq)]


                seq_len = len(seq) + 1
                length_penalty = ((5 + seq_len) ** alpha) / (6 ** alpha)

                top_k = np.argsort(predictions)[-beam_width:]

                for token_id in top_k:
                    new_seq = seq + [token_id]


                    new_score = (score * (len(seq) / seq_len) + np.log(predictions[token_id])) / length_penalty
                    candidates.append((new_seq, new_score))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

            if all(seq[-1] == 1 for seq, _ in beams):
                completed_beams.extend(beams)
                break

        if completed_beams:
            best_seq, _ = max(completed_beams, key=lambda x: x[1])
        else:
            best_seq, _ = max(beams, key=lambda x: x[1])

        caption_words = []
        for token_id in best_seq:
            word = ''
            for w, idx in self.tokenizer.word_index.items():
                if idx == token_id:
                    word = w
                    break
            if word and word != '<UNK>':
                caption_words.append(word)

        return " ".join(caption_words).strip().capitalize() + "."