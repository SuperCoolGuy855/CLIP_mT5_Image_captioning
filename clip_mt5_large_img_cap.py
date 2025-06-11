# %%
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    CLIPModel,
    CLIPVisionModel,
    MT5ForConditionalGeneration,
    CLIPImageProcessorFast,
    MT5Tokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from torchvision.transforms.v2 import (
    Compose,
    RandomVerticalFlip,
    RandomHorizontalFlip,
    RandomGrayscale,
)
from torchinfo import summary
from datasets import Dataset, Image as HFImage
from torch import nn
from PIL import Image
from tqdm import trange
import evaluate
import nltk
import torch
import numpy as np
import json
import os
import gc


# %%
def freeze_model_layers(model):
    """
    Completely prevent any layer from being updated
    """
    for param in model.parameters():
        param.requires_grad = False


# %%
class ImageCaptionConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.clip_name = "facebook/metaclip-h14-fullcc2.5b"
        self.mt5_name = "google/mt5-base"


# %%
class ImageCaptionModel(PreTrainedModel):
    config_class = ImageCaptionConfig

    def __init__(self, config):
        super().__init__(config)
        self.clip = CLIPVisionModel.from_pretrained(config.clip_name)
        self.mt5 = MT5ForConditionalGeneration.from_pretrained(config.mt5_name)

        self.projection = nn.Sequential(
            nn.Linear(self.clip.config.hidden_size, self.mt5.config.d_model),
            nn.GELU(),
            nn.LayerNorm(self.mt5.config.d_model),
            nn.Dropout(0.1), # HYPERPARAMETER to be tuned
        )

        # Freeze the CLIP model
        freeze_model_layers(self.clip)

        # Freeze most bottom layers of MT5 decoder
        freeze_model_layers(self.mt5.decoder.block[:-2])

        self.main_input_name = "pixel_values"

    def get_image_features(self, pixel_values):
        clip_output = self.clip(pixel_values=pixel_values, return_dict=False)[0]
        # normalized_output = self.clip.vision_model.post_layernorm(clip_output) # TODO: Check if this is correct
        casted_output = clip_output.to(torch.float32)
        projection_embed = self.projection(casted_output)
        return projection_embed

    def forward(self, pixel_values, labels):
        image_features = self.get_image_features(pixel_values)
        mt5_output = self.mt5(inputs_embeds=image_features, labels=labels)

        return mt5_output

    def generate(self, pixel_values, max_length=96):
        with torch.no_grad():
            image_features = self.get_image_features(pixel_values)
            mt5_output = self.mt5.generate(
                inputs_embeds=image_features, max_length=max_length
            )

        return mt5_output


# %%
def get_dataset(dataset_folder: str, json_path: str):
    data = json.load(open(json_path, encoding="utf-8"))

    image_map = {}
    for item in data["images"]:
        image_path = os.path.join(dataset_folder, item["filename"])
        image_map[item["id"]] = image_path

    caption_map = {}
    for item in data["annotations"]:
        if item["image_id"] not in caption_map:
            caption_map[item["image_id"]] = []
        caption_map[item["image_id"]].append(item["caption"])

    dataset_dict = {"images": [], "captions": []}
    for image_id, image_path in image_map.items():
        dataset_dict["images"].append(image_path)
        dataset_dict["captions"].append(caption_map[image_id])

    dataset = Dataset.from_dict(dataset_dict)
    casted_dataset = dataset.cast_column("images", HFImage())
    casted_dataset.set_format("pt")

    return casted_dataset


# %%
def get_training_dataset():
    # return get_dataset(r"ktvic_dataset/train-images", r"ktvic_dataset/train_data.json")
    return Dataset.load_from_disk(r"train_dataset")


# %%
config = ImageCaptionConfig()

# %%
# Preprocess the dataset, randomly flipping the images, and tokenizing the captions
tokenizer = MT5Tokenizer.from_pretrained(config.mt5_name)
processor = CLIPImageProcessorFast.from_pretrained(config.clip_name)


def preprocess_function(examples):
    flip = Compose(
        [RandomVerticalFlip(p=0.5), RandomHorizontalFlip(p=0.5), RandomGrayscale(p=0.1)]
    )
    images = flip(torch.tensor(examples["pixel_values"]))
    captions = tokenizer(
        examples["labels"], padding=True, truncation=True, return_tensors="pt"
    )

    return {"pixel_values": images, "labels": captions["input_ids"]}


def preprocess_images(examples):
    pixel_values = processor(examples["images"], return_tensors="pt").pixel_values
    return {"images": pixel_values}


# %%
class CustomMetricCallback(TrainerCallback):
    def __init__(self):
        super().__init__()

        # Get test dataset
        # EDIT: Change the dataset paths to your own dataset paths
        test_dataset = get_dataset(
            "ktvic_dataset/public-test-images", "ktvic_dataset/test_data.json"
        )
        test_dataset = test_dataset.map(preprocess_images, batched=True, batch_size=12)
        test_dataset.set_format("pt")

        # Get test images tensor and move to device
        test_images = test_dataset["images"]

        # Get test captions
        test_captions = test_dataset["captions"]

        self.test_images = test_images
        self.test_captions = test_captions
        self.max_cider_score = np.float64(0.0)

    def on_epoch_end(self, args, state, control, model: ImageCaptionModel, **kwargs):
        # Print total_flos for funnies
        print(f"Total FLOPS: {state.total_flos}")

        # Set model to eval mode
        model.eval()

        # Transfer test images to device
        self.test_images = self.test_images.to(model.device)

        # Generate predictions in batches
        batch_size = 16
        predicted_output = []
        for i in trange(0, len(self.test_images), batch_size):
            batch = self.test_images[i : i + batch_size]
            output = model.generate(batch)
            output = output.cpu().numpy().tolist()
            predicted_output.extend(output)

        predicted_output = tokenizer.batch_decode(
            predicted_output, skip_special_tokens=True
        )

        # Open metric file in append mode
        f = open("metrics.txt", "a")

        # Write the epoch number
        f.write(f"Epoch: {state.epoch}\n")

        try:
            bleu = evaluate.load("bleu")
            results_1 = bleu.compute(
                predictions=predicted_output, references=self.test_captions, max_order=1
            )
            print(results_1)
            f.write(f"BLEU-1: {results_1}\n")
        except Exception:
            print("BLEU-1 metric not available")
            f.write("BLEU-1 metric not available\n")

        # %%
        try:
            results_4 = bleu.compute(
                predictions=predicted_output, references=self.test_captions
            )
            print(results_4)
            f.write(f"BLEU-4: {results_4}\n")
        except Exception:
            print("BLEU-4 metric not available")
            f.write("BLEU-4 metric not available\n")

        # %%
        # EDIT: Uncomment this to enable cider evaluation, only if you have Java installed
        # try:
        #     cider = evaluate.load("Kamichanw/CIDEr")
        #     results = cider.compute(
        #         predictions=predicted_output, references=self.test_captions
        #     )
        #     print(results)
        #     f.write(f"CIDEr: {results}\n")

        #     if results["CIDEr"] > self.max_cider_score:
        #         self.max_cider_score = results["CIDEr"]
        #         # Save the model with epoch number in the name
        #         model.save_pretrained(f"model_epoch_{state.epoch}")
        #         print(f"Model saved with CIDEr score: {self.max_cider_score}")
        # except Exception:
        #     print("CIDEr metric not available")
        #     f.write("CIDEr metric not available\n")

        # %%
        try:
            meteor = evaluate.load("meteor")
            results = meteor.compute(
                predictions=predicted_output, references=self.test_captions
            )
            print(results)
            f.write(f"METEOR: {results}\n")
        except Exception:
            print("METEOR metric not available")
            f.write("METEOR metric not available\n")

        # %%
        try:
            rouge = evaluate.load("rouge")
            results = rouge.compute(
                predictions=predicted_output, references=self.test_captions
            )
            print(results)
            f.write(f"ROUGE: {results}\n")
        except Exception:
            print("ROUGE metric not available")
            f.write("ROUGE metric not available\n")

        # Close the file
        f.close()

        # Set model back to train mode
        model.train()

        # Transfer test images back to cpu
        self.test_images = self.test_images.to("cpu")

        # Scrub VRAM and cache
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    # torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

    # %%
    train_dataset = get_training_dataset()
    # Cut the dataset to 100 samples for testing
    # train_dataset = train_dataset.select(range(100))

    # %%
    train_dataset.set_transform(preprocess_function)

    # %%
    # Create model
    model = ImageCaptionModel(config)

    # Examine the model
    print(model)

    print(summary(model))

    # %%
    # EDIT: Disable tf32 training if not supported
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # %%
    # EDIT: Change the training arguments as needed
    # Training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=20,
        per_device_train_batch_size=32,
        dataloader_num_workers=12,
        # gradient_checkpointing=True,
        bf16=True,
        tf32=True, # EDIT: Disable tf32 training if not supported
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=500,
        # use_cpu = True,
        dataloader_persistent_workers=True,
    )

    # %%
    callbacks = CustomMetricCallback()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=[callbacks],
    )

    # %%
    # Train the model
    trainer.train(resume_from_checkpoint=True)

    # %%
    # Save the model
    # trainer.save_model("model")

    # %%
    del trainer
    del model
    del train_dataset
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
