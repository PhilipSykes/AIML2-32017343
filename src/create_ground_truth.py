import os
import json
from collections import defaultdict


def create_ground_truth_json(captions_file, test_folder, output_file):
    test_images = set(os.listdir(test_folder))

    image_captions = defaultdict(list)

    print("Processing captions file...")
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) == 0:
                continue

            try:
                image_id, caption = line.strip().split(',', 1)
                image_filename = image_id.split('#')[0]

                if image_filename in test_images:
                    caption = caption.strip().strip('.')
                    image_captions[image_filename].append(caption)
            except ValueError:
                print(f"Skipping malformed line: {line.strip()}")

    ground_truth = dict(image_captions)

    print(f"Saving ground truth for {len(ground_truth)} images...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"Ground truth file created at: {output_file}")
    print(f"Number of images with captions: {len(ground_truth)}")

    print("\nSample entries:")
    for image_id, captions in list(ground_truth.items())[:3]:
        print(f"\nImage: {image_id}")
        print("Captions:")
        for caption in captions[:2]:
            print(f"- {caption}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    captions_file = "../data/captions.txt"
    test_folder = "../data/Flickr8k_Dataset/test"
    output_file = "../data/test_captions.json"

    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found at: {captions_file}")
    if not os.path.exists(test_folder):
        raise FileNotFoundError(f"Test folder not found at: {test_folder}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    create_ground_truth_json(captions_file, test_folder, output_file)