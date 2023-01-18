from flask import Flask
from flask import request
from werkzeug.middleware.proxy_fix import ProxyFix
import json
import os
import faiss
from langdetect import detect

import string
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F


# EMB_PATH_KNRM = "data/knrm_emb.bin"
# VOCAB_PATH = "data/vocab.json"
# MLP_PATH = "data/knrm_mlp.bin"
# EMB_PATH_GLOVE = "data/glove.6B.50d.txt"

EMB_PATH_KNRM = os.environ["EMB_PATH_KNRM"]
VOCAB_PATH = os.environ["VOCAB_PATH"]
MLP_PATH = os.environ["MLP_PATH"]
EMB_PATH_GLOVE = os.environ["EMB_PATH_GLOVE"]

class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp( (-(x-self.mu)**2)/(2*self.sigma**2) ) 

class KNRM(torch.nn.Module):
    def __init__(self, embedding_matrix: np.ndarray, freeze_embeddings: bool, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()
        self.embeddings = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix),
            freeze=freeze_embeddings,
            padding_idx=0
        )
        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        K = self.kernel_num
        step = 1 / (K - 1)
        ar_1 = np.linspace(step, 1-step, (K-1)//2, endpoint=True)
        ar_2 = -ar_1[::-1]
        ar = np.hstack((ar_2, ar_1, np.array([1])))
        mu_list = torch.from_numpy(ar)
        sigma_list = [self.sigma] * (K - 1) + [self.exact_sigma]
        kernels = torch.nn.ModuleList([GaussianKernel(mu, sigma) for mu, sigma in zip(mu_list, sigma_list)])
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        K = self.kernel_num
        model = torch.nn.Sequential()
        if len(self.out_layers) > 0:
            for i, count_layers in enumerate(self.out_layers):
                model.add_module(str(i), torch.nn.Linear(K, count_layers))
                model.add_module(torch.nn.ReLU())
            model.add_module(str(i+1), torch.nn.Linear(count_layers[-1], 1))
        else:
            model.add_module('0', torch.nn.Linear(K, 1))
        return model

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:
        embedding_query = self.embeddings(query.long())   
        embedding_doc = self.embeddings(doc.long())
        matching_matrix = torch.einsum('bld,brd->blr',
                                        F.normalize(embedding_query, p=2, dim=-1),
                                        F.normalize(embedding_doc, p=2, dim=-1))
        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        # shape = [Batch, Left, Embedding], [Batch, Right, Embedding]
        query, doc = inputs['query'], inputs['document']

        # shape = [Batch, Left, Right]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape = [Batch, Kernels]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape = [Batch]
        out = self.mlp(kernels_out)
        return out

def collate_fn(batch_objs: List[Dict[str, torch.Tensor]]):
    max_len_q1 = -1
    max_len_d1 = -1
    for elem in batch_objs:
        max_len_q1 = max(len(elem['query']), max_len_q1)
        max_len_d1 = max(len(elem['document']), max_len_d1)

    q1s = []
    d1s = []
    
    for elem in batch_objs:
        left_elem, label = elem
        pad_len1 = max_len_q1 - len(elem['query'])
        pad_len2 = max_len_d1 - len(elem['document'])
        q1s.append(elem['query'] + [0] * pad_len1)
        d1s.append(elem['document'] + [0] * pad_len2)
        
    q1s = torch.LongTensor(q1s)
    d1s = torch.LongTensor(d1s)

    ret_left = {'query': q1s, 'document': d1s}
    return ret_left

class Solution:
    def __init__(self, 
                 freeze_knrm_embeddings: bool = True,
                 num_candidates: int = 10,
                 knrm_out_mlp: List[int] = [],
                 knrm_kernel_num: int = 21,
                 ):     

        self.trans = str.maketrans(string.punctuation, " " * len(string.punctuation))
        self.freeze_knrm_embeddings = freeze_knrm_embeddings
        self.knrm_kernel_num = knrm_kernel_num
        self.knrm_out_mlp = knrm_out_mlp
        self.num_candidates = num_candidates

        self.vocab = self.read_vocab()
        self.model = self.build_knrm_model()
        
        self.embeddings = self.read_glove_embeddings(EMB_PATH_GLOVE)
        self.emb_glove_size = len(list(self.embeddings.values())[-1])
        self.faiss_index = None
        self.idx_to_text_mapping = None # self.get_idx_to_text_mapping(self.glue_train_df)
        
    def read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        corpus = {}
        with open(file_path, "r", encoding="utf-8") as glove_embdgs:
            for line in glove_embdgs:
                word, *vector = line.split()
                corpus[word] = list(map(float, vector))
        return corpus
        
    def read_vocab(self) -> Dict[str, int]:
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return vocab

    def build_knrm_model(self) -> Tuple[torch.nn.Module, Dict[str, int], List[str]]:
        with open(EMB_PATH_KNRM, "rb") as f:
            emb_in_dict_form = torch.load(f)
            
        model = KNRM(emb_in_dict_form["weight"], self.freeze_knrm_embeddings, 
                     out_layers=self.knrm_out_mlp, kernel_num=self.knrm_kernel_num)
        
        with open(MLP_PATH, "rb") as f:
            model.mlp.load_state_dict(torch.load(f))
        return model
    
    def eng_lang_check(self, inp_str: str) -> bool:
        return detect(inp_str) == "en"
    
    def is_faiss_index_initialized(self) -> bool:
        return self.faiss_index is not None

    def hadle_punctuation(self, inp_str: str) -> str:
        return inp_str.translate(self.trans)

    def simple_preproc(self, inp_str: str) -> List[str]:
        return nltk.word_tokenize(self.hadle_punctuation(inp_str.lower().strip()))

    def from_text_to_emb(self, text: str) -> np.array:
        tokens = self.simple_preproc(text)
        embeddings_list = [self.embeddings[token] for token in tokens if token in self.embeddings]

        if embeddings_list:
            text_embedding = np.average(embeddings_list, axis=0)
        else:
            text_embedding = np.zeros(self.count_words_in_glove)
        return text_embedding.astype('float32').reshape(-1)

    def update_index(self, documents: Dict[str, str]) -> int:
        self.idx_to_text_mapping = {key: val for key, val in documents.items()}
        
        idx_list = []
        embeddings_list = []
        for idx, doc in self.idx_to_text_mapping.items():
            doc_emb = self.from_text_to_emb(doc)
            idx_list.append(idx)
            embeddings_list.append(doc_emb)
        idxs = np.stack(idx_list, axis=0).reshape(-1).astype('int64')
        embeddings = np.stack(embeddings_list, axis=0).astype('float32')
        
        faiss_index = faiss.IndexFlatL2(self.emb_glove_size)
        faiss_index = faiss.IndexIDMap(faiss_index)
        faiss_index.add_with_ids(embeddings, idxs)
        self.faiss_index = faiss_index
        return self.faiss_index.ntotal

    def get_candidates(self, query: str) -> np.array:
        embedding = self.from_text_to_emb(query).reshape(1, -1)
        Distances, Indices = self.faiss_index.search(embedding, self.num_candidates)
        return np.array([x for x in Indices.squeeze() if x > -1])[:self.num_candidates]
    
    def tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        tokenized_list = []
        for el in tokenized_text:
            word_in_dict = self.vocab.get(el)
            if word_in_dict:
                tokenized_list.append(word_in_dict)
            else:
                tokenized_list.append(self.vocab['OOV'])
        return tokenized_list

    def convert_text_idx_to_token_idxs(self, idx: str) -> List[int]:
        list_with_cleaned_text = self.simple_preproc(self.idx_to_text_mapping[str(idx)])
        tokenized_text_to_index = self.tokenized_text_to_index(list_with_cleaned_text)
        return tokenized_text_to_index
    
    def ranking_candidates(self, query: str, idx_candidates: np.array) -> np.array:
        candidates_list = []
        tokenized_query = self.tokenized_text_to_index(self.simple_preproc(query))
        for idx in idx_candidates:
            dictionary = {"query": tokenized_query,
                          "document": self.convert_text_idx_to_token_idxs(idx)}
            candidates_list.append(dictionary)
            
        candidates = collate_fn(candidates_list)
        preds = self.model.predict(candidates).squeeze()
        _, idxs = torch.sort(preds, descending=True)
        return idx_candidates[idxs]
    
    def query(self, queries: Dict[str, List[str]]):
        checked_eng_lang_list: List[bool] = list(map(self.eng_lang_check, queries))
        suggestions: List[Optional[List[Tuple[str, str]]]] = []
        for query, is_eng_lang in zip(queries, checked_eng_lang_list):
            if not is_eng_lang:
                suggestions.append(None)
            else:
                idx_candidates = self.get_candidates(query)
                idx_ranked_candidates = self.ranking_candidates(query, idx_candidates)
                suggestion = [(str(idx), self.idx_to_text_mapping[str(idx)]) for idx in idx_ranked_candidates]
                suggestions.append(suggestion)
        return checked_eng_lang_list, suggestions


app = Flask(__name__)
hint_system = None

@app.route("/ping", methods=["GET"])
def ping():
    result = {"status": "ok"}
    return json.dumps(result)

@app.route("/update_index", methods=["POST"])
def update_index():
    global hint_system
    docs: Dict[str, str] = json.loads(request.json)["documents"]
    if hint_system is None:
        hint_system = Solution(num_candidates=10)
    count_docs = hint_system.update_index(docs)
    result = {"status": "ok", "index_size": count_docs}
    return json.dumps(result)

@app.route("/query", methods=["POST"])
def query():
    global hint_system
    if hint_system is None or not hint_system.is_faiss_index_initialized():
        result = {"status": "FAISS is not initialized!"}
        return json.dumps(result)
    queries: Dict[str, List[str]] = json.loads(request.json)["queries"]
    print(queries)
    en_lang_check, suggestions = hint_system.query(queries)
    result = {"lang_check": en_lang_check, "suggestions": suggestions}
    return json.dumps(result)

# app.wsgi_app = ProxyFix(app.wsgi_app)
app.run(debug=False, host="127.0.0.1", port=11000)