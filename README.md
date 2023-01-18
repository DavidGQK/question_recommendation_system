# Question recommendation system for similar questions
The search is done exclusively by the main question. The system is represented by a microservice based on `Flask`. <br/>

`main.py` - file with the Flask-based web service and all the necessary logic <br/>
`glove.6B.50d.txt` - raw `GLOVE` embeddings from the `Quora Question Pairs` dataset. You can easily find it on the Internet <br/>

Files obtained after preprocessing raw embeddings (realization of `KNRM architechture` you can find [here](https://github.com/DavidGQK/KNRM_architecture)): <br/>
- `knrm_mlp.bin` - binary file with `KNRM` top weights for candidate reranking <br/>
- `vocab.json `- JSON wordlist. In it the key is a word, the value is an index in the embedding matrix<br/>
- `knrm_emb.bin` - `KNRM` embedding matrix. The order of the tokens in it corresponds to the JSON wordlist<br/>

## How it works

First, the query is filtered by language (using the `LangDetect library`) - all queries for which a particular language does not equal `"en"` are excluded. Then candidate questions are searched for using `FAISS`. These candidates are reranked by the `KNRM`, and then `up to 10 candidates` are given as answers

`/ping` - for checking readiness

The data come and should be given in json view. You can get them for example like this: <br/>
`content = json.loads(request.json)`, where request is an object from `Flask library`

Then there are two endpoints [for queries (to search for `similar questions`) and to create `FAISS-index`]:

`/query` - accepts `POST` request. Should return json where `status='FAISS is not initialized!'` in case questions were not loaded for search using second method

`Query format:`<br/>
A json, with a single 'queries' key whose value is a list of rows with questions `(Dict[str, List[str]])`

`Response format (in case of a created index):`<br/>
A json with two fields. `lang_check` describes whether the query was recognized as English (`List[bool], True/False-values`), the values are `List[Optional[List[Tuple[str, str]]]]`

In this list for each query from the query a list (`up to 10`) of found similar questions, where each question is represented as a `Tuple`, where the first value is the id, the second is the unprocessed text of the similar question itself. If language check failed (not English) or there was some breakdown in processing - leave `None` in the list instead of answer (e.g. `[[(..., ...), (..., ...), ...], None, ... ]`)

`/update_index` - accepts `POST` request with json field documents, `Dict[str,str]` - all documents, where key is text id, value is text itself. It is implied that initialization is a one-time thing. The returned json has two keys: status (`ok`, if everything went well) and `index_size`, which value is a single integer storing the number of documents in the index
