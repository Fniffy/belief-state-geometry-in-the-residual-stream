import torch
import torch.nn as nn
from constants import *
from pathlib import Path
from parantheses_dataset import CustomParenthesesDataset as Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence



def collate_fn(batch):
    features = [item['features'] for item in batch]
    labels = [item['label'] for item in batch]
    max_len = max(max(f.shape[0] for f in features), max(l.shape[0] for l in labels))
    features_padded = pad_sequence(features, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0)
    # If needed, pad manually to max_len
    if features_padded.shape[1] != max_len:
        features_padded = nn.functional.pad(features_padded, (0, max_len - features_padded.shape[1]))
    if labels_padded.shape[1] != max_len:
        labels_padded = nn.functional.pad(labels_padded, (0, max_len - labels_padded.shape[1]))
    return features_padded, labels_padded

class ResidualParenthesesModel(nn.Module):
    def __init__(self, num_tokens = len(MAPPING), embedding_dimension=EMBEDDING_DIMENSION, num_layers=LAYERS, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embedding_dimension)
        
        # simple residual MLP blocks
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dimension, d_model),
                nn.ReLU(),
                nn.Linear(d_model, embedding_dimension)
            )
            for _ in range(num_layers)
        ])
        self.last_residual_stream = None
        self.fc_out = nn.Linear(embedding_dimension, num_tokens)  # output logits for each token

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        h = x
        h_array = []
        for layer in self.layers:
            h_array.append(h)
            h = h + layer(h)
        h_array.append(h) # last residual stream
        self.last_residual_stream = h_array
        logits = self.fc_out(h)  # [batch, seq_len, num_tokens]
        return logits
    
if __name__ == "__main__":
    if TRAINING:
        
        model = ResidualParenthesesModel()
        if Path(CUT_PARENTHESES_MODEL).exists():
            model.load_state_dict(torch.load(CUT_PARENTHESES_MODEL))
            print("loaded existing model weights")
        criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 is padding
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        dataset = Dataset("belief-state-geometry-in-the-residual-stream/src/data/trainingdata/training_?_paranthesees.txt")
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
        test_dataset = Dataset("belief-state-geometry-in-the-residual-stream/src/data/testdata/test_?_data.txt")
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle = True)
        print(f"loaded Dataset of size: {len(dataset)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.train()

        for epoch in range(EPOCHS):
            running_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)  # [batch, seq_len, num_tokens]
                loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(dataloader):.4f}")
            
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for test_x, test_y in test_dataloader:
                    test_x, test_y = test_x.to(device), test_y.to(device)
                    test_logits = model(test_x)
                    loss = criterion(test_logits.view(-1, test_logits.size(-1)), test_y.view(-1))
                    test_loss += loss.item()
            print(f"Test Loss: {test_loss/len(test_dataloader):.4f}")
            model.train()

        torch.save(model.state_dict(), CUT_PARENTHESES_MODEL)
        print("Model saved.")   
        
    else:
        model = ResidualParenthesesModel()
        if Path(CUT_PARENTHESES_MODEL).exists():
            model.load_state_dict(torch.load(CUT_PARENTHESES_MODEL))
            print("loaded existing model weights")
        else:
            print("No model weights found, exiting.")
            exit(1)
        model.eval()
        while True:
            try:
                user_input = input("Enter parentheses everything else exits the loop: ")
                parentheses_string = user_input.strip()+ '_'*(MAX_STRING_LENGTH*2 - len(user_input.strip()))
                parentheses_vector = torch.tensor([MAPPING[c] for c in parentheses_string], dtype=torch.float)
                output = model(parentheses_vector.unsqueeze(0).long())
                # After output = model(parentheses_vector.unsqueeze(0).long())
                pred_indices = output.argmax(dim=-1).squeeze().tolist()  # list of predicted token indices
                # Convert indices back to characters using inverse mapping
                INV_MAPPING = {v: k for k, v in MAPPING.items()}
                predicted_string = ''.join([INV_MAPPING[idx] for idx in pred_indices if INV_MAPPING[idx] == ')'])
                print(f"Predicted closing parentheses: {predicted_string}")
            except ValueError:
                print("eval phase stopped")
                break