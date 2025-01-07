import os
import pytest
from PIL import Image
import matplotlib.pyplot as plt
from .caption_generator import CaptionGenerator


def load_tokenizer(tokenizer_path):
    import pickle
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)


@pytest.fixture
def tokenizer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tokenizer_path = os.path.join(current_dir, "models", "tokenizer", "tokenizer.pkl")
    return load_tokenizer(tokenizer_path)


def test_generate_caption(tokenizer):
    print("\nTokenizer Details:")
    print(f"Vocabulary size: {len(tokenizer.word_index) + 1}")
    print("Sample tokens:", list(tokenizer.word_index.items())[:10])
    print("Special tokens:", [word for word in tokenizer.word_index.keys()
                              if word.startswith('<') and word.endswith('>')])

    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "fold_1_best_model.keras")
    test_image_path = os.path.join(current_dir, "test_images", "sample_image.jpg")

    assert os.path.exists(model_path), f"Model not found at {model_path}"
    assert os.path.exists(test_image_path), f"Test image not found at {test_image_path}"

    generator = CaptionGenerator(
        model_path=model_path,
        tokenizer=tokenizer
    )

    try:
        caption = generator.generate_caption(test_image_path)
        print("\nGenerated Caption:", caption)

        assert caption is not None, "Caption should not be None"
        assert isinstance(caption, str), "Caption should be a string"
        assert len(caption) > 0, "Caption should not be empty"

        plt.figure(figsize=(10, 8))
        img = Image.open(test_image_path)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Generated Caption: {caption}", pad=20)
        plt.show()

    except Exception as e:
        pytest.fail(f"Caption generation failed with error: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])