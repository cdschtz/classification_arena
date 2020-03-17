from torchtext import data
from torchtext.vocab import GloVe

from datasets import WikiSyntheticGeneral


def load_dataset(dataset_name='WikiSyntheticGeneral', splits=None, tokenize_func=lambda x: x.split(),
                 embedding_func='glove'):
    """
    Arguments:
        dataset_name: which dataset to use, for training (and testing) use WikiSyntheticGeneral,
            then with trained models try out WikiSyntheticSophisticated or WikiNews
        splits: ratio of training, test and validation split, e.g. [0.7, 0.15, 0.15]
        tokenize_func: function to use for tokenization, e.g. split() or 'spacy'
        embedding_func: which embedding function to use, can be 'glove' or 'fasttext'
    """
    # text_fields will receive (batches of) Strings of text
    text_field = data.Field(sequential=True, tokenize=tokenize_func, lower=True, batch_first=True)
    # label_fields will receive the (numerical) labels of data points (e.g. 0, 1), i.e. use_vocab=False
    label_field = data.LabelField(use_vocab=False)

    if dataset_name == 'WikiSyntheticGeneral':
        dataset = WikiSyntheticGeneral(text_field, label_field)
    else:
        raise ValueError(f"Error: dataset_name {dataset_name} is not valid!")

    if splits is None:
        splits = [0.7, 0.15, 0.15]
    else:
        splits = [0.7, 0.15, 0.15]
    train_data, test_data, valid_data = dataset.split(splits)

    emb_func = None
    if embedding_func == 'glove':
        emb_func = GloVe(name='6B', dim=300)

    # TODO: use pre-built vocab files instead??
    text_field.build_vocab(dataset.text, vectors=emb_func)
    del dataset

    word_embeddings = text_field.vocab.vectors
    print("Length of Text Vocabulary: " + str(len(text_field.vocab)))
    print("Vector size of Text Vocabulary: ", text_field.vocab.vectors.size())

    train_iter, test_iter, valid_iter = data.Iterator.splits(
        (train_data, test_data, valid_data), batch_size=32, sort_key=lambda x: len(x.text), repeat=False, shuffle=True
    )

    vocab_size = len(text_field.vocab)

    return text_field, vocab_size, word_embeddings, train_iter, test_iter, valid_iter
