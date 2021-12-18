import torch
import torch.nn as nn
import numpy as np
import time
import math
from model import TransAm
from plot import plot_and_loss, predict_future
from getdata import get_data, get_batch, create_inout_sequences


input_window = 5*24  # number of input steps
output_window = 1   # number of prediction steps, in this model its fixed to one
batch_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


torch.manual_seed(0)
np.random.seed(0)




# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512))     # (S,N,E)
# tgt = torch.rand((20, 32, 512))     # (T,N,E)
# out = transformer_model(src, tgt)


def train(train_data):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data)//batch_size, scheduler.get_lr()[0], elapsed*1000/log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()



def evaluate(eval_model, data_source):
    """
    :param eval_model: model
    :param data_source: input data
    :return: return loss to evaluate model
    """
    eval_model.eval()   # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 1000
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data)            
            total_loss += len(data[0]) * criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


train_data, val_data = get_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.005
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


best_val_loss = float("inf")
epochs = 10
best_model = None

train_loss_all, val_loss_all = [], []


for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    train_loss = evaluate(model, train_data)
    val_loss = evaluate(model, val_data)
    train_loss_all.append(train_loss)
    val_loss_all.append(val_loss)
    if epoch % 10 == 0:
        val_loss = plot_and_loss(model, val_data, epoch)
        predict_future(model, val_data, 200)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} | train ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), train_loss, math.exp(train_loss)))
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(
        epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)

    # if val_loss < best_val_loss:
    #    best_val_loss = val_loss
    #    best_model = model

    scheduler.step() 

print("train_loss_all:", train_loss_all, len(train_loss_all))
print("val_loss_all:", val_loss_all, len(val_loss_all))


import matplotlib.pyplot as plt
import numpy as np
plt.plot(np.array(train_loss_all), label='train')
plt.plot(np.array(val_loss_all), label='validation')
plt.legend()
plt.show()


# src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
# out = model(src)
# print(out)
# print(out.shape)