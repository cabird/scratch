
# code adapted from https://towardsdatascience.com/semantic-similarity-using-transformers-8f3cb5bf66d6
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
import numpy as np

import pandas
import collections
import math

Response = collections.namedtuple("Response", "response codes embedding")
Embedding = collections.namedtuple("Embedding", "sentence embedding")

#roberta base
# 148
# [44, 64, 76]

def main():
    
    model_name =  "roberta-base-nli-stsb-mean-tokens"
    model_name = "stsb-roberta-large"
    print(f"loading {model_name} model")

    model = SentenceTransformer(model_name)
    #tokenizer = RobertaTokenizer.from_pretrained(model_name)
    #model = RobertaModel.from_pretrained(model_name)
    file = "ListOfOKRsForMachineLearning.xlsx"
    print(f"loading {file}")
    df1 = pandas.read_excel(file)
    df2 = pandas.read_excel(file, 1)

    print(df1["Objectives"])
    print(df2["Key Results"])

    sentences = []
    for index, row in df1.iterrows():
        sentences.append(row[0])

    for index, row in df2.iterrows():
        sentences.append(row[0])

    print(sentences)
    embeddings = []

    print("calculating embeddings")
    for sentence in sentences:
        embedding = model.encode(sentence, convert_to_tensor=True)
        embeddings.append(Embedding(sentence, embedding))

    for sent1 in embeddings:
        best_similarity = -1
        best_embedding = None
        for sent2 in embeddings:
            similarity = util.pytorch_cos_sim(sent1.embedding, sent2.embedding)
            if similarity > best_similarity:
                best_embedding = sent2
                best_similarity = similarity
        print(f" Sentence 1: {sent1.sentence}")
        print(f" Best sentence: {best_embedding.sentence} {best_similarity}")
            

          

def example():
    model = SentenceTransformer('stsb-roberta-large')
    
    sentence1 = "I like Python because I can build AI applications"
    sentence2 = "I like Python because I can do data analytics"
    
    # encode sentences to get their embeddings
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)
    
    # compute similarity scores of two embeddings
    cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)
    print("Sentence 1:", sentence1)
    print("Sentence 2:", sentence2)
    print("Similarity score:", cosine_scores.item())

if __name__ == "__main__":
    main()