import pandas as pd
import numpy as np
from tqdm import trange
from transformers import pipeline

BATCH_SIZE = 2048


if __name__ == '__main__':
    data = pd.read_csv('data.csv')
    data['full_joke'] = data['set-up'] + ' ' + data['punchline']
    data['label'] = ['' for i in range(len(data))]

    labels = []
    classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device='cuda:0')
    candidate_labels = ["politics", "neutral", "racist", "offending"]
    for i in trange((len(data['full_joke']) + BATCH_SIZE - 1) // BATCH_SIZE):
        inputs = list(data.iloc[i*BATCH_SIZE:(i+1)*BATCH_SIZE]['full_joke'])
        outputs = classifier(inputs, candidate_labels, multi_label=False)
        labels.append([output['labels'][np.argmax(output['scores'])] for output in outputs])

    data['label'] = np.concatenate(labels)
    data.to_csv('labeled.csv')

