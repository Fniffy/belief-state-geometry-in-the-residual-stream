import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

def get_hf_tokenizer():
    if not os.path.exists("./char_tokenizer"):
        
        print("Training new tokenizer...")

        dataset = load_dataset('csv', data_files='belief-state-geometry-in-the-residual-stream/src/data/trainingdata/training_data.csv', delimiter=',')
        print(dataset)
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
            if row.get('feature'):
                texts.append(row['feature'])
            if row.get('label'):
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
    else:
        hf_tokenizer = PreTrainedTokenizerFast.from_pretrained("./char_tokenizer")
        
if __name__ == "__main__":
    hf_tokenizer = get_hf_tokenizer()
    print("Tokenizer is ready.")