import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, LSTM, Embedding, Dropout, LayerNormalization,
    TimeDistributed, Concatenate, GlobalAveragePooling2D, RepeatVector,
    Reshape
)


class ImageCaptionModel:

    def __init__(self, vocab_size, max_length, embedding_dim=512, units=1024):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.units = units

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.compile_model()

    def _build_encoder(self):
        input_layer = Input(shape=(256, 256, 3))
        base_model = ResNet50V2(weights='imagenet', include_top=False, input_tensor=input_layer)

        for layer in base_model.layers[:-60]:
            layer.trainable = False

        x = base_model.output

        x = tf.keras.layers.Reshape((-1, 2048))(x)

        attention = Dense(512, activation='tanh')(x)
        attention_weights = Dense(1, activation='softmax', kernel_initializer='zeros')(attention)
        x = tf.keras.layers.Multiply()([x, attention_weights])

        x = tf.keras.layers.GlobalAveragePooling1D()(x)

        x = Dense(1024, activation='relu')(x)
        x = LayerNormalization()(x)
        x = Dropout(0.4)(x)

        x = Dense(768, activation='relu')(x)
        x = LayerNormalization()(x)
        x = Dropout(0.4)(x)


        output = Dense(self.embedding_dim, activation='relu', name='encoder_output')(x)

        return Model(inputs=input_layer, outputs=output)

    def _build_decoder(self):

        image_features_input = Input(shape=(self.embedding_dim,))
        caption_input = Input(shape=(self.max_length - 1,))


        embed = Embedding(
            self.vocab_size,
            self.units,
            mask_zero=False
        )(caption_input)


        image_features = Dense(self.units, activation='relu')(image_features_input)
        image_features = RepeatVector(self.max_length - 1)(image_features)


        decoder_input = Concatenate(axis=2)([image_features, embed])


        x = LSTM(self.units, return_sequences=True)(decoder_input)
        x = LayerNormalization()(x)
        x = Dropout(0.4)(x)

        x = LSTM(self.units, return_sequences=True)(x)
        x = LayerNormalization()(x)
        x = Dropout(0.4)(x)


        x = Dense(self.units * 2, activation='relu')(x)
        x = Dropout(0.4)(x)


        output = TimeDistributed(Dense(self.vocab_size, activation='softmax'))(x)

        return Model(inputs=[image_features_input, caption_input], outputs=output)

    def compile_model(self):
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )

        self.decoder.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )

    def get_config(self):
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'embedding_dim': self.embedding_dim,
            'units': self.units
        }