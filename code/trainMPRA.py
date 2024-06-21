#!/usr/bin/env python
# coding: utf-8

# In[1]:


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


# In[2]:


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


# In[3]:


def checkWeight(model):
    first_layer = list(model.children())[0]
    # Check if it has learnable parameters (weights)
    if hasattr(first_layer, 'weight'):
        first_layer_weight = first_layer.weight.data
        print("First layer weight tensor:", first_layer_weight[0])
    else:
        print("The first layer does not have learnable parameters (weights).")


# In[1]:


def trainModel(trainData_seqs,valData_seqs, model_savename,BATCH_SIZE,INIT_LR,early_stop_thresh,EPOCHS,verbose, num_kernels=(128, 256), dropout_rate=0.1):
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_accuracy = -1
    best_epoch = -1
    trainDataLoader = DataLoader(trainData_seqs, shuffle=True,
        batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(valData_seqs, batch_size=BATCH_SIZE)

    # calculate steps per epoch for training and validation set
    trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
    valSteps = len(valDataLoader.dataset) // BATCH_SIZE

    #print('trainSteps and valSteps: ',[trainSteps, valSteps])
    # for (x, y) in trainDataLoader:
    #     print(x)

    
    #print("[INFO] initializing the deep learning model...")

    #model = DeepSEAlight_single().to(device)
    #model = CNN_single1().to(device)
    model = CNN_single1(num_kernels=num_kernels, dropout_rate=dropout_rate).to(device)

    # initialize our optimizer and loss function
    opt = Adam(model.parameters(), lr=INIT_LR)

    lossfun = nn.NLLLoss()
    #lossfun= nn.BCEWithLogitsLoss()

    # initialize a dictionary to store training history
    H = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # measure how long training is going to take
    #print("[INFO] training the network...")
    startTime = time.time()

    model.train()
    #print(model)
    

    for e in range(0, EPOCHS):
        model.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalValLoss = 0
        # initialize the number of correct predictions in the training and validation step
        trainCorrect = 0
        valCorrect = 0
        # loop over the training set
        for (x, y) in trainDataLoader:
            # send the input to the device
            (x, y) = (x.to(device,dtype=torch.float), y.to(device))
            # perform a forward pass and calculate the training loss
            pred = model(x)
            loss = lossfun(pred, y)
            # zero out the gradients, perform the backpropagation step,
            # and update the weights
            opt.zero_grad()
            loss.backward()
            opt.step()
            # add the loss to the total training loss so far and
            # calculate the number of correct predictions
            totalTrainLoss += loss
            trainCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

        # switch off autograd for evaluation
        with torch.no_grad():
            # set the model in evaluation mode
            model.eval()
            # loop over the validation set
            for (x, y) in valDataLoader:
                (x, y) = (x.to(device,dtype=torch.float), y.to(device))
                # make the predictions and calculate the validation loss
                pred = model(x)
                totalValLoss += lossfun(pred, y)
                # calculate the number of correct predictions
                valCorrect += (pred.argmax(1) == y).type(
                    torch.float).sum().item()

        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgValLoss = totalValLoss / valSteps
        # calculate the training and validation accuracy
        trainCorrect = trainCorrect / len(trainDataLoader.dataset)
        valCorrect = valCorrect / len(valDataLoader.dataset)
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["train_acc"].append(trainCorrect)
        H["val_loss"].append(avgValLoss.cpu().detach().numpy())
        H["val_acc"].append(valCorrect)

 
        # print the model training and validation information
        #if EPOCHS%5==1:
        #    print("[INFO] EPOCH: {}/{}".format(e + 1, EPOCHS))
        #    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        #        avgTrainLoss, trainCorrect))
        #    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        #        avgValLoss, valCorrect))


        if valCorrect > best_accuracy:
            best_accuracy = valCorrect
            best_epoch = e
            checkpoint(model, model_savename)
        elif e - best_epoch > early_stop_thresh:
            #print("Early stopped training at epoch %d" % e)
            break  # terminate the training loop
            
    if verbose==1:
        get_ipython().run_line_magic('matplotlib', 'inline')
        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label="train_loss")
        plt.plot(H["val_loss"], label="val_loss")
        plt.plot(H["train_acc"], label="train_acc")
        plt.plot(H["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
    
    # finish measuring how long training took
    endTime = time.time()
    #print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))
    return model


# In[8]:


def eval_model(preds,predsProb,y_test,verbose=0):    
    y_test_prob = predsProb
    y_test_classes=preds
    
    # ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
    auc_test = auc(fpr, tpr)
    # Precision-Recall Curve and AUPRC
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_test_prob)
    auprc_test = auc(recall, precision)
    
    acc_test=accuracy_score(y_test_classes, y_test)
    f1_test = f1_score(y_test_classes, y_test, average='binary')
    recall_test = recall_score(y_test_classes, y_test, average='binary')
    precision_test = precision_score(y_test_classes, y_test, average='binary')
    R_test=pearsonr(y_test, y_test_prob)[0]
    
    acc_test=round(acc_test,3)
    auc_test=round(auc_test,3)
    auprc_test = round(auprc_test, 3)
    f1_test=round(f1_test,3)
    precision_test=round(precision_test,3)
    recall_test=round(recall_test,3)
    R_test=round(R_test,3)
    
    if verbose==1:
        get_ipython().run_line_magic('matplotlib', 'inline')
        print(f'Test: acc {acc_test:.3f}, auc {auc_test:.3f}, auprc {auprc_test:.3f}, f1 {f1_test:.3f}, precision {precision_test:.3f}, recall {recall_test:.3f}, R {R_test:.3f}\n')
        #print('Test: acc %.3f, auc %.3f, f1 %.3f, precision %.3f, recall %.3f,R %.3f\n' % (acc_test, auc_test, f1_test, precision_test, recall_test,R_test))
        #plt.figure()
        #lw = 2
        #plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc_test)
        #plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        #plt.xlim([0.0, 1.0])
        #plt.ylim([0.0, 1.05])
        #plt.xlabel('False Positive Rate')
        #plt.ylabel('True Positive Rate')
        #plt.title('ROC')
        #plt.legend(loc="lower right")
        #plt.show()
    return [acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test]


# In[1]:


def testModel(model,model_savename,testData_seq,BATCH_SIZE,verbose=0, predonly = 0):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testDataLoader = DataLoader(testData_seq, batch_size=BATCH_SIZE)

    #print("[INFO]  resume best model...")
    
    resume(model, model_savename)

    # we can now evaluate the network on the test set
    #print("[INFO] evaluating network...")
    # turn off autograd for testing evaluation
    with torch.no_grad():
        # set the model in evaluation mode
        model.eval()

        # initialize a list to store our predictions
        preds = []
        predsProb = []
        ys = []

        # loop over the test set
        for (x, y) in testDataLoader:
            # send the input to the device
            x = x.to(device,dtype=torch.float)
            # make the predictions and add them to the list
            pred = model(x)
            preds.extend(pred.argmax(axis=1).cpu().numpy())
            predsProb.extend(pred[:,1].cpu().numpy())
            ys.extend(y.cpu().numpy())


    preds=np.array(preds)
    predsProb=np.array(predsProb)
    ys=np.array(ys)
    
    if predonly == 0:
        acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test = eval_model(preds,predsProb,ys,verbose=verbose)
        return [acc_test, auc_test, auprc_test, f1_test, precision_test, recall_test, R_test, predsProb, preds, y_test]
    else:
        return predsProb
    


# In[ ]:




