import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import BartTokenizer
from constants import *
from datasets import load_dataset
from transformers import BartConfig, BartForConditionalGeneration, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

dataset = load_dataset('csv', data_files='belief-state-geometry-in-the-residual-stream/src/data/testdata/test_?_paranthesees.csv', delimiter=',')

# Initialize character-level WordLevel tokenizer
tokenizer = Tokenizer(models.WordLevel(unk_token="[UNK]"))
trainer = trainers.WordLevelTrainer(
    special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
)

# Character-level pre-tokenizer (splits into characters)
tokenizer.pre_tokenizer = pre_tokenizers.Split("", "isolated")  

# Collect all strings (features + labels) for training
texts = []
for row in dataset['train']:
    texts.append(row['feature'])
    texts.append(row['label'])

tokenizer.train_from_iterator(texts, trainer=trainer)

# Wrap in HF tokenizer
hf_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]"
)

hf_tokenizer.save_pretrained("./char_tokenizer")

def preprocess(batch):
    model_inputs = hf_tokenizer(
        batch["feature"], max_length=128, truncation=True, padding="max_length"
    )
    labels = hf_tokenizer(
        batch["label"], max_length=128, truncation=True, padding="max_length"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = dataset.map(preprocess, batched=True)

config = BartConfig(
    vocab_size=len(hf_tokenizer),
    d_model=256,               # small hidden size
    encoder_layers=3,
    decoder_layers=3,
    encoder_attention_heads=4,
    decoder_attention_heads=4,
    max_position_embeddings=512
)

model = BartForConditionalGeneration(config)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./bart-char-scratch",
    evaluation_strategy="epoch",
    learning_rate=5e-4,   # higher LR since training from scratch
    per_device_train_batch_size=16,
    num_train_epochs=30,
    weight_decay=0.01,
    logging_steps=20,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    tokenizer=hf_tokenizer
)

trainer.train()
