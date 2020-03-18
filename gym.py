import time
import datetime
import torch
import torch.nn.functional as F

import load_data
from models.category_2 import RNN, LSTMClassifier
from utils import DEVICE, clip_gradient, plot_train_data, write_results


def get_model(name, model_config):
    output_size = 2  # binary classification of text
    batch_size = 32 if 'batch_size' not in model_config else model_config['batch_size']
    hidden_size = 256 if 'hidden_size' not in model_config else model_config['hidden_size']
    embedding_length = 300 if 'embedding_length' not in model_config else model_config['embedding_length']
    word_embeddings = model_config['word_embeddings']
    vocab_size = model_config['vocab_size']

    if name == 'rnn':
        return RNN(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)
    elif name == 'lstm':
        return LSTMClassifier(batch_size, output_size, hidden_size, vocab_size, embedding_length, word_embeddings)


def train_model(model, iters, num_epochs, batch_size, loss_fn, optim, test=False):
    start_time = datetime.datetime.now()
    time_info = {"start": start_time.isoformat(timespec='seconds')}

    all_train_losses = all_train_accuracies = []
    train_loss = train_acc = test_loss = test_acc = 0.0  # calculated at end of last epoch
    train_iter, val_iter, test_iter = iters

    for epoch in range(num_epochs):
        total_epoch_loss = 0
        total_epoch_acc = 0

        model.train()
        for idx, batch in enumerate(train_iter):
            text = batch.text.to(DEVICE)
            target = batch.label.to(DEVICE)
            if text.shape[0] is not batch_size:
                continue

            optim.zero_grad()
            prediction = model(text)
            loss = loss_fn(prediction, target)
            all_train_losses.append(loss.item())

            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
            acc = 100.0 * num_corrects / len(batch)
            all_train_accuracies.append(acc.item())

            loss.backward()
            clip_gradient(model, 1e-1)
            optim.step()

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

        train_loss = total_epoch_loss / len(train_iter)
        train_acc = total_epoch_acc / len(train_iter)

    val_loss, val_acc = evaluate_model(model, val_iter, loss_fn, batch_size)
    if test:
        test_loss, test_acc = evaluate_model(model, val_iter, loss_fn, batch_size)

    end_time = datetime.datetime.now()
    time_elapsed = end_time - start_time
    time_info.update({
        "end": end_time.isoformat(timespec='seconds'),
        "duration": '0' + str(time_elapsed)[:7]  # pad hours and remove milliseconds
    })

    loss_info = {"train": train_loss, "val": val_loss, "test": test_loss}
    acc_info = {"train": train_acc, "val": val_acc, "test": test_acc}
    graph_data = {"losses": all_train_losses, "accuracies": all_train_accuracies}

    return acc_info, loss_info, graph_data, time_info


def evaluate_model(model, val_iter, loss_fn, batch_size):
    loss = acc = 0.0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text.to(DEVICE)
            target = batch.label.to(DEVICE)
            if text.shape[0] is not batch_size:
                continue

            prediction = model(text)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            loss += loss.item()
            acc += acc.item()

    return loss/len(val_iter), acc/len(val_iter)


def run():
    dataset_name = "WikiSyntheticGeneral"
    splits = [0.7, 0.15, 0.15]
    tokenize_fn = 'split'
    embedding_fn = 'glove'
    vocabulary = 'built'
    batch_size = 32
    hidden_size = 256
    vocab_size, word_embeddings, iters, sizes = load_data.\
        load_dataset(dataset_name=dataset_name, splits=splits, tokenize_func=tokenize_fn, embedding_func=embedding_fn,
                     batch_size=batch_size)

    model_name = 'lstm'
    model_config = {
        "word_embeddings": word_embeddings,
        "vocab_size": vocab_size,
        "batch_size": batch_size,
        "hidden_size": hidden_size
    }
    model = get_model('rnn', model_config)
    model.to(DEVICE)

    num_epochs = 3
    loss_function = F.cross_entropy
    learning_rate = 1e-3  # adam default: 1e-3, proposed: 2e-5
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    acc_info, loss_info, graph_data, time_info = train_model(
        model, iters, num_epochs, batch_size, loss_function, optimizer
    )
    # plot_train_data(losses[::10], accuracies[::10])

    data_obj = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "splits": splits,
        "sizes": sizes,
        "tokenizer": tokenize_fn,
        "embeddings": embedding_fn,
        "vocabulary": vocabulary,
        "vocab_size": vocab_size,
        "epochs": num_epochs,
        "learn_rate": learning_rate,
        "batch_size": batch_size,
        "hidden_size": hidden_size,
        "train_acc": acc_info['train'],
        "val_acc": acc_info['val'],
        "test_acc": acc_info['test'],
        "losses": graph_data['losses'],
        "accuracies": graph_data['accuracies'],
        "device": DEVICE.type,
        "time_info": time_info
    }
    write_results(data_obj)


def run_all():
    pass


if __name__ == '__main__':
    run()
