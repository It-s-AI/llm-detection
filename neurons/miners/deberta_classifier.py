import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from torch.utils.data import Dataset
from tqdm import tqdm


class SimpleTestDataset(Dataset):
    def __init__(self, strings, tokenizer, max_sequence_length):
        self.Strings = strings
        self.Tokenizer = tokenizer
        self.MaxSequenceLength = max_sequence_length

    def __len__(self):
        return len(self.Strings)

    def __getitem__(self, idx):
        string = self.Strings[idx].strip()
        token_ids = self.Tokenizer(string, max_length=self.MaxSequenceLength, truncation=True).input_ids

        return {
            'input_ids': token_ids,
        }


def GeneratePredictions(model, tokenizer, test_dataset, device):
    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=1,
        collate_fn=DataCollatorWithPadding(tokenizer))

    all_predictions = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            token_sequences = batch.input_ids.to(device)
            attention_masks = batch.attention_mask.to(device)

            with torch.cuda.amp.autocast():
                raw_predictions = model(token_sequences, attention_masks).logits

            scaled_predictions = raw_predictions.softmax(dim = 1)[:,1]
            all_predictions.append(scaled_predictions.cpu().numpy())

    all_predictions = np.concatenate(all_predictions)

    return all_predictions


class DebertaClassifier:
    def __init__(self, foundation_model_path, model_path, device):
        self.tokenizer = AutoTokenizer.from_pretrained(foundation_model_path)
        self.max_length = 1024
        self.device = device

        model = AutoModelForSequenceClassification.from_pretrained(
            foundation_model_path,
            state_dict=torch.load(model_path),
            attention_probs_dropout_prob=0,
            hidden_dropout_prob=0).to(device)

        self.model = model.eval()

    def predict_batch(self, texts):
        test_dataset = SimpleTestDataset(texts, self.tokenizer, self.max_length)
        return GeneratePredictions(self.model, self.tokenizer, test_dataset, self.device)

    def __call__(self, text):
        return self.predict_batch([text])[0]
