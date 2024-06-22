#!/usr/bin/env python
# coding: utf-8

# In[1]:


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


# In[2]:
def resume(model, filename):
    model.load_state_dict(torch.load(filename))


# In[1]:

def trainModel(trainData_seqs,valData_seqs, model_savename,BATCH_SIZE,INIT_LR,early_stop_thresh,EPOCHS, num_kernels=(128, 256), dropout_rate=0.1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_accuracy = -1
    best_epoch = -1
    trainDataLoader = DataLoader(trainData_seqs, shuffle=True,
        batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData_seqs, batch_size=BATCH_SIZE)

    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE


    model = CNN_single1(num_kernels=num_kernels, dropout_rate=dropout_rate).to(device)

    opt = Adam(model.parameters(), lr=INIT_LR)

    lossfun = nn.NLLLoss()

    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    model.train()
    
    for e in range(0, EPOCHS):
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0
        for (x, y) in trainDataLoader:
            (x, y) = (x.to(device,dtype=torch.float), y.to(device))
            pred = model(x)
            loss = lossfun(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()


        with torch.no_grad():

            model.eval()
            for (x, y) in valDataLoader:
                (x, y) = (x.to(device,dtype=torch.float), y.to(device))
                pred = model(x)
                totalValLoss += lossfun(pred, y)
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

        if valCorrect > best_accuracy:
            best_accuracy = valCorrect
            best_epoch = e
            checkpoint(model, model_savename)
        elif e - best_epoch > early_stop_thresh:
            break
            
    return model

