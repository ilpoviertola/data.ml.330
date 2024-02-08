# Meeting 06.02.2024

Present: Ilpo, Valtteri and Timo

Agenda: Divide tasks for Operation Transformer project

## ONN Resources

- <https://pypi.org/project/pygop/>
- <https://github.com/junaidmalik09/fastonn/tree/master/fastonn>

## Route map for the project

1. Find a dataset and some benchmark metric for it + tokenizer
    - Can be anything
2. Build a basic simple transformer model
    - PyTorch
    - MinGPT
3. Replace prediction head with operational neurons.

## Tasks

1. Find a dataset (book-theme) (Timo)
   - [List of datasets](datasets.md)
2. Find a metric to evalaute the quality of the model (Valtteri)
3. Find a tokenizer for the dataset (Ilpo)
    - <https://huggingface.co/docs/transformers/en/main_classes/tokenizer>
4. (Build a simple transformer model) (FUTURE)

### Tokenizer (Ilpo)

I think byte-pair encoding (BPE) would be a good choice for the tokenizer. It's a subword tokenization algorithm that is widely used in NLP. It's also used in the GPT-2 model, so it would be a good fit for our project. It is fairly simple to implement (actually it is already implemented in the minGPT code). We can reuse that or then use one from the Hugging Face library (<https://huggingface.co/docs/tokenizers/en/quicktour>). It is also quick to train once we have decided the dataset we are going to use. OR we can use the GPT-2 tokenizer directly (<https://github.com/openai/tiktoken>).

### Testing
https://aclanthology.org/2021.eacl-main.202.pdf

1. Word-overlap metrics such as BLEU (n =
1,2,3,4), METEOR and ROUGE (n = 1,2)
presented low correlation with human ratings.
2. BERTScore and BLEURT outperformed other
metrics and produced higher correlation with
human ratings than other metrics on all
diagrams. BERTScore values range between
[0.23, 0.43] and for BLEURT values range
between [0.26, 0.53].
3. Human ratings for Informativeness and
Clarity are highly correlated with each other,
as observed in Figure 2 (r = 0.82)

