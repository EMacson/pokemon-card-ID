from datasets import Dataset, load_dataset
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from PIL import Image
import os, json

TRAIN_FOLDER = "data\\train"
TEST_FOLDER = "data\\test"

def json_to_prompt(json_data):
    prompt = "<s_pokemon_card>"
    for key, value in json_data.items():
        if key in ["attacks", "abilities"]:
            prompt += f"<{key}>"
            for attack in value:
                prompt += "<item>"
                for subkey, subval in attack.items():
                    prompt += f"<{subkey}>{subval}</{subkey}>"
                prompt += "</item>"
            prompt += f"</{key}>"
        else:
            prompt += f"<{key}>{value}</{key}>"
    prompt += "</s_pokemon_card>"
    return prompt

def load_examples(folder):
    examples = []
    for file in os.listdir(folder):
        if file.endswith('.json'):
            imageFile = os.path.join(folder, file.replace('.json', '.jpg'))
            jsonFile = os.path.join(folder, file)

            with open(jsonFile, 'r', encoding='utf-8') as f:
                data = json.load(f)

            image = Image.open(imageFile).convert('RGB')
            label = json_to_prompt(data)
            #print(label, "\n\n")

            examples.append({"image": image, "label": label})

    return examples

print("TRAIN DATASET")
train_dataset = Dataset.from_list(load_examples(TRAIN_FOLDER))

print("\nTEST DATASET")
test_dataset = Dataset.from_list(load_examples(TEST_FOLDER))

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base")

# Update special tokens
model.decoder.resize_token_embeddings(len(processor.tokenizer))

def preprocess(examples):
    # Resize image to a manageable size
    image = examples["image"].resize((480, 360), Image.BILINEAR)

    pixel_values = processor(image, return_tensors="pt").pixel_values[0]
    decoder_input_ids = processor.tokenizer("<s_pokemon_card>", add_special_tokens=False).input_ids
    labels = processor.tokenizer(examples["label"], add_special_tokens=False).input_ids
    return {
        "pixel_values": pixel_values,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels
    }

train_dataset = train_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

args = Seq2SeqTrainingArguments(
    output_dir="./donut-pokemon",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=10,
    evaluation_strategy="steps",
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    save_total_limit=2,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()