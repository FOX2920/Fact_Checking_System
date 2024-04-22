# Importing necessary libraries
import math
import re

# Data handling libraries
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# PyTorch libraries
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
# NLP libraries
import nltk
nltk.download('punkt')

# Multithreading libraries
from multiprocessing import Pool, cpu_count

class BM25:
    def __init__(self, corpus, tokenizer=None):
        self.corpus_size = 0
        self.avgdl = 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.tokenizer = tokenizer

        if tokenizer:
            corpus = self._tokenize_corpus(corpus)

        nd = self._initialize(corpus)
        self._calc_idf(nd)

    def _initialize(self, corpus):
        nd = {}  # word -> number of documents with word
        num_doc = 0
        for document in corpus:
            self.doc_len.append(len(document))
            num_doc += len(document)

            frequencies = {}
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                try:
                    nd[word]+=1
                except KeyError:
                    nd[word] = 1

            self.corpus_size += 1

        self.avgdl = num_doc / self.corpus_size
        return nd

    def _tokenize_corpus(self, corpus):
        pool = Pool(cpu_count())
        tokenized_corpus = pool.map(self.tokenizer, corpus)
        return tokenized_corpus

    def _calc_idf(self, nd):
        raise NotImplementedError()

    def get_scores(self, query):
        raise NotImplementedError()

    def get_batch_scores(self, query, doc_ids):
        raise NotImplementedError()

    def get_top_n(self, query, documents, n=5):
        assert self.corpus_size == len(documents), "The documents given don't match the index corpus!"

        scores = self.get_scores(query)
        min_score = np.min(scores)
        max_score = np.max(scores)

        # Scale scores to 0-1 range
        if max_score != min_score:
            scaled_scores = (scores - min_score) / (max_score - min_score)
        else:
            scaled_scores = np.ones(self.corpus_size)

        top_n_indices = np.argsort(scaled_scores)[::-1][:n]
        top_n_scaled_scores = [scaled_scores[i] for i in top_n_indices]

        return [documents[i] for i in top_n_indices], top_n_scaled_scores


class BM25Okapi(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        """
        Calculates frequencies of terms in documents and in corpus.
        This algorithm sets a floor on the idf values to eps * average_idf
        """
        # collect idf sum to calculate an average idf for epsilon value
        idf_sum = 0
        # collect words with negative idf to set them a special epsilon value.
        # idf can be negative if word is contained in more than half of documents
        negative_idfs = []
        for word, freq in nd.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        self.average_idf = idf_sum / len(self.idf)

        eps = self.epsilon * self.average_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query):
        """
        The ATIRE BM25 variant uses an idf function which uses a log(idf) score. To prevent negative idf scores,
        this algorithm also adds a floor to the idf value of epsilon.
        See [Trotman, A., X. Jia, M. Crane, Towards an Efficient and Effective Search Engine] for more info
        :param query:
        :return:
        """
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score

    def get_batch_scores(self, query, doc_ids):
        """
        Calculate bm25 scores between query and subset of all docs
        """
        assert all(di < len(self.doc_freqs) for di in doc_ids)
        score = np.zeros(len(doc_ids))
        doc_len = np.array(self.doc_len)[doc_ids]
        for q in query:
            q_freq = np.array([(self.doc_freqs[di].get(q) or 0) for di in doc_ids])
            score += (self.idf.get(q) or 0) * (q_freq * (self.k1 + 1) /
                                               (q_freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)))
        return score.tolist()
    
def preprocess_text(text: str) -> str:
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text

def evidence_top_n(context, query):
    sentences = split_text(context)
    tokenized_sentences = [str(doc).split(" ") for doc in sentences]
    bm25 = BM25Okapi(tokenized_sentences)
    tokenized_query = query.split(" ")
    top_docs, top_scores = bm25.get_top_n(tokenized_query, sentences, n=5)

    return top_docs, top_scores

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def similarities(context: list, text: str, weight: list):
    sentences = [text] + context
    tokenizer_sbert = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1')
    encoded_input = tokenizer_sbert(sentences, padding=True, truncation=True, return_tensors='pt')
    encoded_input = {key: value.to('cuda') for key, value in encoded_input.items()}
    model_sbert = AutoModel.from_pretrained('sentence-transformers/paraphrase-xlm-r-multilingual-v1').to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model_sbert(**encoded_input)
    # Perform pooling. In this case, mean pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    similarities = []
    claim_embeddings = sentence_embeddings[0].unsqueeze(0)
    for i in (range(1, len(sentence_embeddings))):
        evidence_embeddings = sentence_embeddings[i].unsqueeze(0)
        cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
        similarity = cosine(claim_embeddings.to(device), evidence_embeddings.to(device)).item()
        # scaled_similarity = ((similarity + 1) / 2) * weight[i-1]
        similarities.append((sentences[i], similarity))

    simi_values = [s[1] for s in similarities]
    scaler = MinMaxScaler()
    scaled_simi_values = scaler.fit_transform(np.array(simi_values).reshape(-1, 1)).flatten()
    similarities = [(sentences[i+1], scaled_value * weight[i]) for i, scaled_value in enumerate(scaled_simi_values)]

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = [item[0] for item in similarities[:1]]
    simi = [item[1] for item in similarities[:1]]
    top_5 = [item[0] for item in similarities[:5]]
    return top_k, simi, top_5

def clean_quotes(sentence):
    # Replace characters within quotes
      return re.sub(r'"([^"]*)"', lambda m: m.group(0).replace('!', '').replace(',', '').replace('?', ''), sentence)

def remove_brackets(text):
    return re.sub(r'\([^)]*\)', lambda m: m.group(0).replace('...', '').replace('.', ''), text)

def split_text(content):
    # Split the text by "\n\n"
    paragraphs = content.split('\n\n')

    # Split each paragraph into sentences
    sentences = []
    for paragraph in paragraphs:
        paragraph = paragraph.replace('...)', ')')
        paragraph = paragraph.replace('... ,', ',')
        paragraph = re.sub(r'\.\.\.(?=\")', '', paragraph)
        paragraph = paragraph.replace('\n', ' ')  # Remove internal line breaks
        paragraph = clean_quotes(paragraph)
        paragraph = re.sub(r'\.(\s[a-z])', lambda match: match.group(1).upper(), paragraph)
        paragraph = paragraph.replace(' .', '.')  # Remove space before period
        paragraph = re.sub(r'\?(?=\s+[a-z])', ' ', paragraph)
        paragraph = re.sub(r'\.\.\.(?=\,)', '', paragraph)
        paragraph = re.sub(r'\.\.\.(?=\s+[a-z])', ' ', paragraph)
        paragraph = paragraph.replace('...', '. ')  # Replace "..." with ". "
        paragraph = paragraph.replace('..', '. ')  # Replace ".." with ". "
        paragraph = paragraph.replace('. ', ' . ')  # Add space after period
        paragraph = paragraph.replace('  ', ' ')  # Remove extra spaces
        paragraph = paragraph.strip()  # Strip leading/trailing spaces

        # Tokenize the paragraph into sentences using NLTK
        paragraph_sentences = nltk.sent_tokenize(paragraph)
        sentences.extend(paragraph_sentences)

    return sentences




