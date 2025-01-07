import pickle
import csv
import os
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer(captions_file):

    captions = []

    with open(captions_file, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)
        for row in csv_reader:
            if len(row) >= 2:
                caption = f"<start> {row[1].strip()} <end>"
                captions.append(caption)

    if not captions:
        raise ValueError("No captions found in file")

    tokenizer = Tokenizer(num_words=5000, oov_token="<UNK>")
    tokenizer.fit_on_texts(captions)

    return tokenizer


def save_tokenizer(tokenizer, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(tokenizer, f)


def main():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    captions_file = os.path.join(project_root, "data", "captions.txt")
    tokenizer_path = os.path.join(script_dir, "models", "tokenizer", "tokenizer.pkl")


    tokenizer = create_tokenizer(captions_file)
    save_tokenizer(tokenizer, tokenizer_path)


    print(f"\nVocabulary size: {len(tokenizer.word_index)}")
    print("\nSpecial tokens:")
    special_tokens = {k: v for k, v in tokenizer.word_index.items()
                      if k.startswith('<') and k.endswith('>')}
    print(special_tokens)

    print(f"\nTokenizer saved to: {tokenizer_path}")


if __name__ == "__main__":
    main()