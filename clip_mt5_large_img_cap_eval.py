import os
import json
import pandas as pd
import gc
from time import sleep

import torch
import torch.nn as nn

from datasets import Dataset, Image as HuggingFaceImage
import nltk
import evaluate
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    TrainingArguments,
    Trainer,
    CLIPModel,
    CLIPProcessor,
    PreTrainedModel,
    PretrainedConfig,
    TrainerCallback
)


def freeze_model_layers(model):
    """
    Completely prevent any layer from being updated
    """
    for param in model.parameters():
        param.requires_grad = False


class CLIPMT5ImageCaptioningConfig(PretrainedConfig):
    model_type = "image_captioning"

    def __init__(
        self,
        clip_model_name="openai/clip-vit-base-patch32",
        mt5_model_name="google/mt5-small",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.clip_model_name = clip_model_name
        self.mt5_model_name = mt5_model_name


class CLIPMT5ImageCaptioningModel(PreTrainedModel):
    config_class = CLIPMT5ImageCaptioningConfig

    def __init__(self, config):
        super().__init__(config)
        self.clip = CLIPModel.from_pretrained(config.clip_model_name)
        # self.clip_preprocess = CLIPProcessor.from_pretrained(config.clip_model_name)
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(config.mt5_model_name)
        self.tokenizer = MT5Tokenizer.from_pretrained(config.mt5_model_name)

        clip_output_dim = self.clip.config.projection_dim
        mt5_input_dim = self.mt5.config.d_model
        self.projection = nn.Linear(clip_output_dim, mt5_input_dim)

        # Freeze CLIP
        freeze_model_layers(self.clip)

        # Freeze MT5
        # freeze_model_layers(self.mt5)

    def forward(self, images, captions):
        # Encode images using CLIP
        image_features = self.clip.get_image_features(images)
        image_embeddings = self.projection(image_features)

        # Prepare inputs for MT5
        # labels = self.tokenizer(captions, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_caption_length)
        outputs = self.mt5(
            inputs_embeds=image_embeddings.unsqueeze(1),
            labels=captions,
        )
        return {
            "loss": outputs.loss,
            # "logits": outputs.logits,
        }

    def generate(self, images):
        with torch.no_grad():
            image_features = self.clip.get_image_features(images)
            image_embeddings = self.projection(image_features)
            outputs = self.mt5.generate(inputs_embeds=image_embeddings.unsqueeze(1))
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # EDIT: Change this to the path of the dataset
    dataset_root = "ktvic_dataset"

    # Prepare test data
    nltk.download("wordnet")
        
    with open(f"{dataset_root}/test_data.json", encoding="utf8") as f:
        test_data = json.load(f)

    image_metadata = test_data["images"]
    annotations = test_data["annotations"]

    image_index = {
        image["id"]: (f"{dataset_root}/public-test-images/{image['filename']}", [])
        for image in image_metadata
    }

    for x in annotations:
        # image_name = image_index[x["image_id"]]["filename"]
        # image_paths.append(f"{dataset_root}/public-test-images/{image_name}")
        # captions.append(x["caption"])
        image_index[x["image_id"]][1].append(x["caption"])

    image_paths = []
    captions = []

    for image_path, caption_list in image_index.values():
        image_paths.append(image_path)
        captions.append(caption_list)

    test_dict = {
        "images": image_paths,
        "captions": captions,
    }

    test_df = pd.DataFrame(test_dict)
    test_dataset = Dataset.from_pandas(test_df).cast_column(
        "images", HuggingFaceImage()
    )
    print(test_dataset)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    def test_transforms(example_batch):
        preprocess_image = processor(
            images=example_batch["images"], padding=True, return_tensors="pt"
        )
        return {
            "images": preprocess_image.pixel_values,
            "captions": example_batch["captions"],
        }

    test_preprocess_dataset = test_dataset.map(
        test_transforms, batched=True, batch_size=16
    )
    test_preprocess_dataset.set_format("torch")
    print(test_preprocess_dataset)

    test_images = test_preprocess_dataset["images"]
    test_captions = test_preprocess_dataset["captions"]

    # Load pre-trained model
    model = CLIPMT5ImageCaptioningModel.from_pretrained("mt5_large_img_cap") # EDIT: Change this to the path of the model

    predicted_output = []

    for i in range(0, len(test_images), 16):
        batch_images = test_images[i : i + 16]
        batch_images = batch_images.to(device)
        batch_output = model.generate(batch_images)
        predicted_output.extend(batch_output)
        del batch_images
    # print(predicted_output)

    bleu = evaluate.load("bleu")
    results_1 = bleu.compute(
        predictions=predicted_output, references=test_captions, max_order=1
    )
    print(results_1)

    results_4 = bleu.compute(predictions=predicted_output, references=test_captions)
    print(results_4)

    meteor = evaluate.load("meteor")
    results = meteor.compute(predictions=predicted_output, references=test_captions)
    print(results)

    rouge = evaluate.load("rouge")
    results = rouge.compute(predictions=predicted_output, references=test_captions)
    print(results, flush=True)

    # EDIT: Uncomment this to enable cider evaluation, only if you have Java installed
    #cider = evaluate.load("Kamichanw/CIDEr")
    #results = cider.compute(predictions=predicted_output, references=test_captions)
    #print(results)
