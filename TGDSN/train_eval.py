import torch
from utils import scoring_func
denorm=130

    
def train_TGAT(model, train_dl, optimizer, criterion, config, device,data_id):
    model.train()
    epoch_loss = 0
    epoch_score = 0


    for inputs, labels in train_dl:
        adj_path = "D:/TGDSN/data/FD00{}_adj.pt".format(data_id)
        my_dataset_adj = torch.load(adj_path)
        src = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        pred, _ ,_= model(my_dataset_adj,src)
        pred  = pred * denorm
        labels = labels * denorm
        rul_loss = criterion(pred.squeeze(), labels)
        score = scoring_func(pred.squeeze() - labels)
        rul_loss.backward()
        epoch_loss =epoch_loss + rul_loss.item()
        epoch_score =epoch_score + score
    return epoch_loss / len(train_dl), epoch_score, pred, labels


def evaluate_TGAT(model, test_dl, criterion, config,device,data_id):
    adj_path = "D:/TGDSN/data/FD00{}_adj.pt".format(data_id)
    my_dataset_adj = torch.load(adj_path)
    model.eval()
    epoch_loss = 0
    epoch_score = 0
    predicted_rul = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_dl:
            src = inputs.to(device)
            pred,hidden ,output= model(my_dataset_adj,src)
            pred = pred * denorm
            if labels.max() <= 1:
                labels = labels * denorm
            rul_loss = criterion(pred.squeeze(), labels)
            score = scoring_func(pred.squeeze() - labels)
            epoch_loss =epoch_loss + rul_loss.item()
            epoch_score =epoch_score + score
            predicted_rul =predicted_rul + (pred.squeeze().tolist())
            true_labels =true_labels + labels.tolist()

    model.train()
    return epoch_loss / len(test_dl), epoch_score,predicted_rul,true_labels,hidden ,output

