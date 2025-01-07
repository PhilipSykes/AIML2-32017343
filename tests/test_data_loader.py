import unittest
import sys
from pathlib import Path

print("dataloader test file is being loaded")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


print(f"Python path now includes: {sys.path[0]}")
print(f"Looking for data_loader.py in: {project_root / 'src'}")

try:
    from src.data_loader import DataLoader
    print("Successfully imported DataLoader")
except ImportError as e:
    print(f"Import error details: {e}")


class TestDataLoader(unittest.TestCase):    
    def setUp(self):
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.data_path = os.path.join(project_root, "data", "Flickr8k_Dataset")
        self.captions_file = os.path.join(project_root, "data", "captions.txt")
        self.loader = DataLoader(self.data_path, self.captions_file)
    
    def test_caption_loading(self):
        # Test that captions are loaded correctly
        captions_dict = self.loader.load_captions()
        
        self.assertGreater(len(captions_dict), 0, "No captions were loaded")
        sample_image = list(captions_dict.keys())[0]
        self.assertIsInstance(captions_dict[sample_image], list, 
                            "Captions should be stored in a list")
        sample_caption = captions_dict[sample_image][0]
        self.assertTrue(sample_caption.startswith('startseq'), 
                       "Captions should start with 'startseq'")
        self.assertTrue(sample_caption.endswith('endseq'), 
                       "Captions should end with 'endseq'")

    def test_tokenizer_creation(self):
        # Test the tokenizer is created and configured

        captions_dict = self.loader.load_captions()
        tokenizer, max_length = self.loader.create_tokenizer(captions_dict)

        self.assertIsNotNone(tokenizer, "Tokenizer should not be None")
        self.assertGreater(len(tokenizer.word_index), 2, 
                          "Vocabulary size seems too small")
        
        self.assertGreater(max_length, 0, "Max length should be positive")
        self.assertLess(max_length, 100, "Max length seems unreasonably large")

    def test_image_processing(self):
        # Test that images are loaded and preprocessed correctly
        captions_dict = self.loader.load_captions()
        sample_image = list(captions_dict.keys())[0]
        
        processed_img = self.loader.load_image(sample_image)
        
        self.assertEqual(processed_img.shape[1:3], (224, 224), 
                        "Image should be resized to 224x224")

        self.assertTrue(processed_img.min() >= -255 and processed_img.max() <= 255,
                       "Image values should be in the expected range")

if __name__ == '__main__':
    unittest.main()