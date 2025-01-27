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

# Callback for evaluation after each epoch
class EvaluationCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
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
    
        self.test_images = test_preprocess_dataset["images"]
        self.test_captions = test_preprocess_dataset["captions"]

    def on_epoch_end(self, args, state, control, model, **kwargs):
        # model = kwargs["model"]
        model.eval()
    
        predicted_output = []

        for i in range(0, len(self.test_images), 16):
            batch_images = self.test_images[i : i + 16]
            batch_images = batch_images.to(device)
            batch_output = model.generate(batch_images)
            predicted_output.extend(batch_output)
            del batch_images
        # print(predicted_output)
    
        bleu = evaluate.load("bleu")
        results_1 = bleu.compute(
            predictions=predicted_output, references=self.test_captions, max_order=1
        )
        print(results_1)
    
        results_4 = bleu.compute(predictions=predicted_output, references=self.test_captions)
        print(results_4)
    
        meteor = evaluate.load("meteor")
        results = meteor.compute(predictions=predicted_output, references=self.test_captions)
        print(results)
    
        rouge = evaluate.load("rouge")
        results = rouge.compute(predictions=predicted_output, references=self.test_captions)
        print(results, flush=True)

        # EDIT: Uncomment this to enable cider evaluation, only if you have Java installed
        #cider = evaluate.load("Kamichanw/CIDEr")
        #results = cider.compute(predictions=predicted_output, references=self.test_captions)
        #print(results)

        model.train()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    
    # EDIT: Change this to the path of the dataset
    dataset_root = "ktvic_dataset"

    # Prepare train dataset
    with open(f"{dataset_root}/train_data.json", encoding="utf8") as f:
        train_data = json.load(f)

    image_metadata = train_data["images"]
    annotations = train_data["annotations"]

    image_index = {image["id"]: image for image in image_metadata}

    image_paths = []
    captions = []

    for x in annotations:
        image_name = image_index[x["image_id"]]["filename"]
        image_paths.append(f"{dataset_root}/train-images/{image_name}")
        captions.append(x["caption"])

    train_dict = {
        "images": image_paths,
        "captions": captions,
    }

    train_df = pd.DataFrame(train_dict)
    train_dataset = Dataset.from_pandas(train_df).cast_column(
        "images", HuggingFaceImage()
    )
    print(train_dataset)

    config = CLIPMT5ImageCaptioningConfig(
        clip_model_name="openai/clip-vit-base-patch32",
        mt5_model_name="google/mt5-large",
        max_caption_length=192,
    )

    processor = CLIPProcessor.from_pretrained(config.clip_model_name)
    tokenizer = MT5Tokenizer.from_pretrained(config.mt5_model_name)
    
    # Preprocess the images in the dataset
    def process_image(example_batch):
        # images = [x for x in example_batch["images"]]
        # captions = example_batch["captions"]
        preprocess_image = processor(
            images=example_batch["images"], padding=True, return_tensors="pt"
        )
        # labels = tokenizer(
        #     example_batch["captions"],
        #     return_tensors="pt",
        #     padding="max_length",
        #     truncation=True,
        #     max_length=config.max_caption_length,
        # )
        # labels = tokenizer(example_batch["captions"], return_tensors="pt", padding=True)
        return {
            "images": preprocess_image.pixel_values,
            "captions": example_batch["captions"],
        }
    
    # Transform the captions into tokenized form
    # Doing a separate preprocessing step for the captions to make the batched captions length equal
    def text_transform(example_batch):
        labels = tokenizer(
            example_batch["captions"],
            return_tensors="pt",
            padding=True,
        )
        return {
            "images": torch.tensor(example_batch["images"]),
            "captions": labels.input_ids,
        }

    train_preprocess_dataset = train_dataset.map(
        process_image, batched=True, batch_size=16, num_proc=6
    )
    train_preprocess_dataset.set_format("torch")
    train_preprocess_dataset.set_transform(text_transform)
    print(train_preprocess_dataset[0]["images"].shape)
    print(train_preprocess_dataset[0]["captions"].shape)

    # train_preprocess_dataset = train_dataset.with_transform(transforms)
    # # train_preprocess_dataset.set_format("torch")
    # print(train_preprocess_dataset)

    # Create the model
    model = CLIPMT5ImageCaptioningModel(config)

    # Print the number of parameters
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    mt5_p_count = sum(p.numel() for p in model.mt5.parameters() if p.requires_grad)
    print(mt5_p_count)

    clip_p_count = sum(p.numel() for p in model.clip.parameters() if p.requires_grad)
    print(clip_p_count)

    projection_p_count = sum(
        p.numel() for p in model.projection.parameters() if p.requires_grad
    )
    print(projection_p_count)

    total_p_count = mt5_p_count + clip_p_count + projection_p_count
    print(total_p_count)

    os.environ["WANDB_DISABLED"] = "true"

    # EDIT: Disable tf32 training if not supported
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # EDIT: Change the training arguments as needed
    training_args = TrainingArguments(
        output_dir="./mt5_large_results",
        num_train_epochs=20,
        per_device_train_batch_size=64,
        dataloader_num_workers=12,
        # gradient_checkpointing=True,
        bf16=True,
        tf32=True, # EDIT: Disable tf32 training if not supported
        logging_strategy="epoch",
        save_strategy="epoch",
        report_to=None,
        dataloader_persistent_workers=True,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_preprocess_dataset, callbacks=[EvaluationCallback]
    )

    # Start fine-tuning
    trainer.train()

    trainer.save_model("clip_mt5_large_model")
