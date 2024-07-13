import re
from PIL import Image

import torch
import pandas
import numpy as np
from statistics import mode
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from collections import Counter

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

class NewVQA(torch.utils.data.Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform  # 画像の前処理
        self.image_dir = image_dir  # 画像ファイルのディレクトリ
        self.df = pandas.read_json(df_path)  # 画像ファイルのパス，question, answerを持つDataFrame
        self.answer = answer

        # answer
        self.label_encoder = LabelEncoder()

        if self.answer:
            all_answers = [process_text(single_answer['answer']) for multi_answer in self.df['answers'] for single_answer in multi_answer]
            unique_answers = sorted(set(all_answers))
            encoded_answers = self.label_encoder.fit_transform(unique_answers)
            self.answer2id = {answer: encoded for answer, encoded in zip(unique_answers, encoded_answers)}

            def encode_answers(multi_answer):
                return [self.answer2id [process_text(single_answer['answer'])] for single_answer in multi_answer]

            self.df['answers'] = self.df['answers'].apply(encode_answers)

    def question_to_ids(self, tokens):
        return [self.word_dict.get(token, 0) for token in tokens]


    def update_dict(self, dataset):
        self.label_encoder = dataset.label_encoder

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        image = self.transform(image)
        question = tokenizer(text=self.df['question'][idx], padding='max_length',
                            truncation=True, max_length=20, return_tensors='pt')
        question['input_ids'] = question['input_ids'].squeeze(0)
        question['attention_mask'] = question['attention_mask'].squeeze(0)
        question['token_type_ids'] = question['token_type_ids'].squeeze(0)

        if self.answer:
            answers = self.df["answers"][idx]
            mode_answer_idx = mode(answers)
            return image, question, torch.LongTensor(answers), int(mode_answer_idx)

        else:
            return image, question

    def __len__(self):
        return len(self.df)

def process_text(text):
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10'
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r'(?<!\d)\.(?!\d)', '', text)

    # 冠詞の削除
    text = re.sub(r'\b(a|an|the)\b', '', text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't", "isnt": "isn't", "arent": "aren't", "wont": "won't",
        "cant": "can't", "wouldnt": "wouldn't", "couldnt": "couldn't"
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", ' ', text)

    # 句読点をスペースに変換
    text = re.sub(r'\s+,', ',', text)

    # 連続するスペースを1つに変換
    text = re.sub(r'\s+', ' ', text).strip()

    return text