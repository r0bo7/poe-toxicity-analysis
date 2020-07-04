# https://github.com/Noixas/BERT-Toxicity

import numpy as np
import torch.utils.data
import torch
import json
import regex as re
import csv
from datetime import datetime
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification

BATCH_SIZE = 256
MAX_SEQUENCE_LENGTH = 128

def convert_lines(texts, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in texts:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(['[CLS]']+tokens_a+['[SEP]'])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


def get_predictions(comments):
    model.eval()
    with torch.no_grad():
        valid_preds_pews = np.zeros((len(comments)))
        valid_pews = torch.utils.data.TensorDataset(torch.tensor(comments, dtype=torch.long))
        valid_loader_pews = torch.utils.data.DataLoader(valid_pews, batch_size=BATCH_SIZE, shuffle=False)
        for i, (x_batch,) in enumerate(valid_loader_pews):
            pred_pews = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)
            valid_preds_pews[i*BATCH_SIZE:(i+1)*BATCH_SIZE] = pred_pews[:,0].detach().cpu().squeeze().numpy()
        test_df_pews = torch.sigmoid(torch.tensor(valid_preds_pews)).numpy()
        return test_df_pews

device = torch.device('cuda')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=1)
model.load_state_dict(torch.load('bert_pytorch.bin'))
model.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def process_json(name):
    print('Processing', name)
    with open('json/' + name + '.json') as f:
        lines = f.readlines()

    date_to_threads = {}

    for line in lines:
        data = json.loads(line)
        text = data['title']
        if 'selftext' in data:
            text += ' ' + data['selftext']
        date = datetime.utcfromtimestamp(data['created_utc']).strftime('%Y-%m-%d')
        if date not in date_to_threads:
            date_to_threads[date] = []
        date_to_threads[date].append(text)

    with open('csv/' + name + '_texts.csv', 'a') as texts_csv_file, open('csv/' + name + '_output.csv', 'a') as output_csv_file:
        texts_writer = csv.writer(texts_csv_file, delimiter=',')
        output_writer = csv.writer(output_csv_file, delimiter=',')
        dates = [date for date, _ in date_to_threads.items()]
        dates.sort(reverse=True)
        for date in dates:
            texts = date_to_threads[date]
            converted = convert_lines(texts, MAX_SEQUENCE_LENGTH, tokenizer)
            predictions = get_predictions(converted)
            mean_prediction = np.mean(predictions)
            toxic_amount = len([tox for tox in predictions if tox > 0.5])
            total_threads = len(predictions)
            print(date, str(total_threads), str(toxic_amount), str(mean_prediction))
            for i in range(len(texts)):
                texts_writer.writerow([date, re.sub(r'[\r\n\t;]', ' ', texts[i])[0:MAX_SEQUENCE_LENGTH], str(predictions[i])])
            output_writer.writerow([date, str(total_threads), str(toxic_amount), str(mean_prediction)])

process_json('pathofexile')
process_json('diablo3')
process_json('Wolcen')
process_json('Warframe')
process_json('leagueoflegends')
process_json('GlobalOffensive')
process_json('DotA2')
process_json('Overwatch')
process_json('VALORANT')
