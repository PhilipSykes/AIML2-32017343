import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from src.data_loader import DataLoader
from src.model import ImageCaptionModel


class Trainer:
    def __init__(self, data_dir, captions_file, batch_size=16, epochs=50, initial_learning_rate=1e-4, max_length=40):
        self.batch_size = batch_size
        self.epochs = epochs
        self.initial_learning_rate = initial_learning_rate
        self.max_length = max_length

        self.data_loader = DataLoader(data_dir, captions_file, max_length)
        self.captions_dict = self.data_loader.captions_dict
        self.tokenizer = self.data_loader.tokenizer

        vocab_size = len(self.tokenizer.word_index) + 1
        self.model = ImageCaptionModel(vocab_size, self.max_length)
        self.embedding_dim = self.model.encoder.output.shape[1]

        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)

    def _create_lr_schedule(self):
        warmup_steps = 1000
        decay_steps = 10000

        def custom_schedule(step):
            if step < warmup_steps:
                return self.initial_learning_rate * (step / warmup_steps)
            else:
                step = step - warmup_steps
                decay_progress = tf.minimum(step / decay_steps, 1.0)
                cosine_decay = 0.5 * (1 + tf.cos(np.pi * decay_progress))
                return self.initial_learning_rate * cosine_decay

        return custom_schedule

    def _compile_model_with_clip(self):
        initial_learning_rate = self.initial_learning_rate
        decay_steps = 1000

        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=initial_learning_rate,
            first_decay_steps=decay_steps,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=1e-5,
            clipnorm=1.0
        )

        self.model.decoder.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

    def create_dataset(self, data_dict):
        image_ids = list(data_dict.keys())

        def generator():
            while True:
                np.random.shuffle(image_ids)
                for image_id in image_ids:
                    img = self.data_loader._load_image(image_id)
                    img_resized = tf.image.resize(img, (256, 256))

                    if tf.random.uniform(()) > 0.5:
                        img_resized = tf.image.random_brightness(img_resized, 0.2)
                    if tf.random.uniform(()) > 0.5:
                        img_resized = tf.image.random_contrast(img_resized, 0.8, 1.2)

                    caption = np.random.choice(data_dict[image_id])
                    cap_sequence = self.tokenizer.texts_to_sequences([caption])[0]
                    cap_padded = tf.keras.preprocessing.sequence.pad_sequences(
                        [cap_sequence], maxlen=self.max_length, padding='post', truncating='post'
                    )[0]

                    input_sequence = cap_padded[:-1]
                    target_sequence = cap_padded[1:]
                    target_one_hot = tf.one_hot(target_sequence, depth=self.data_loader.vocab_size)

                    image_features = self.model.encoder.predict(tf.expand_dims(img_resized, axis=0))
                    image_features = np.reshape(image_features, (self.embedding_dim,))

                    yield (image_features, input_sequence), target_one_hot

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.embedding_dim,), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.max_length - 1,), dtype=tf.int32),
                ),
                tf.TensorSpec(shape=(self.max_length - 1, self.data_loader.vocab_size), dtype=tf.float32),
            )
        )

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def train_with_kfold(self, k=5):
        image_ids = list(self.captions_dict.keys())
        kfold = KFold(n_splits=k, shuffle=True)
        histories = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(image_ids)):
            print(f'Training fold {fold + 1}/{k}')

            train_ids = [image_ids[i] for i in train_idx]
            val_ids = [image_ids[i] for i in val_idx]

            train_data = {k: self.captions_dict[k] for k in train_ids}
            val_data = {k: self.captions_dict[k] for k in val_ids}

            train_dataset = self.create_dataset(train_data)
            val_dataset = self.create_dataset(val_data)

            self._compile_model_with_clip()

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    f'models/fold_{fold + 1}_best_model.keras',
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True,
                    min_delta=0.0005
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=4,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=f'logs/fold_{fold + 1}',
                    update_freq='epoch',
                    histogram_freq=1
                )
            ]

            steps_per_epoch = len(train_data) // self.batch_size
            validation_steps = len(val_data) // self.batch_size

            history = self.model.decoder.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                epochs=self.epochs,
                validation_data=val_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks
            )

            histories.append(history.history)

        return histories

    def plot_training_results(self, histories):
        metrics = ['loss', 'accuracy', 'top_k_categorical_accuracy']
        plt.style.use('seaborn')

        for fold, history in enumerate(histories):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for i, metric in enumerate(metrics):
                axes[i].plot(history[metric], label=f'Training {metric}')
                axes[i].plot(history[f'val_{metric}'], label=f'Validation {metric}')
                axes[i].set_title(f'Fold {fold + 1} - {metric.capitalize()}')
                axes[i].set_xlabel('Epochs')
                axes[i].set_ylabel(metric.capitalize())
                axes[i].legend()
                axes[i].grid(True)

            plt.tight_layout()
            plt.savefig(f'logs/fold_{fold + 1}_training_results.png', dpi=300, bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    dataset_path = "../data/Flickr8k_Dataset"
    captions_file = "../data/captions.txt"

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(f"Training data size: {len(captions_file)}")

    trainer = Trainer(dataset_path, captions_file)
    histories = trainer.train_with_kfold(k=5)
    print(f"Training completed. History: {histories}")

    trainer.plot_training_results(histories)
