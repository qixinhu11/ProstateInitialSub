import os
import time 

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def cls_score(pred, label):
    """
    pred:  [array[1, 0], array[0, 1]]
    label: [array[1, 0], array[0, 1]]
    """
    pred  = np.argmax(np.array(pred), axis=1)
    label = np.argmax(np.array(label), axis=1)

    accuracy = accuracy_score(label, pred)
    precision = precision_score(label, pred)
    recall = recall_score(label, pred)
    f1 = f1_score(label, pred)

    score_table = {
        "case_acc": accuracy,
        "case_recall": recall,
        "case_precision": precision,
        "case_f1": f1
    }
    score = 0.6 * recall +  0.4 * accuracy

    return score, score_table


def trainer(model, train_loader, val_loader, optimizer, scheduler, loss_function, args):
    device = args.device    
    save_root = os.path.join(args.log_dir, args.modality ,args.model_name)
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    
    best_metric = -1
    for epoch in range(args.max_epochs):
        scheduler.step()
        epoch_time = time.time()
        ## Training
        print("-" * 10)
        model.train()
        epoch_loss = 0
        step = 0
        start_time = time.time()
        for batch_data in train_loader:
            step += 1
            images  = batch_data['image'].to(device)
            labels  = batch_data['label'].to(device)
            optimizer.zero_grad()
            # outputs = model(liver, spleen, left_kidney, right_kidney)
            outputs = model(images)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            outputs  = outputs.detach().cpu().numpy()
            labels   = labels.detach().cpu().numpy()
            print('Epoch {}/{} {}/{}'.format(epoch + 1, args.max_epochs, step, len(train_loader)),
                  'loss: {:.4f}'.format(loss.item()), 
                  'time {:.2f}s'.format(time.time() - start_time))
            with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                print('Epoch {}/{} {}/{}'.format(epoch + 1, args.max_epochs, step, len(train_loader)),
                    'loss: {:.4f}'.format(loss.item()),
                    'time {:.2f}s'.format(time.time() - start_time), file=f)
            start_time = time.time()

        epoch_loss /= step
        print('Final training  {}/{}'.format(epoch + 1, args.max_epochs), 'loss: {:.4f}'.format(epoch_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))
        with open(os.path.join(save_root, 'log.txt'), 'a') as f:
            print('Final training  {}/{}'.format(epoch + 1, args.max_epochs), 'loss: {:.4f}'.format(epoch_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time), file=f)
        
        # save the model
        torch.save(model.state_dict(), os.path.join(save_root, 'model_last.pt'))

        ## Evaluation
        if (epoch + 1) % args.val_every == 0:
            torch.save(model.state_dict(), os.path.join(save_root, f'model_{epoch}.pt'))
            model.eval()
            print("-" * 20)
            print("Start to Evaluation")
            y_true = []
            y_pred = []
            eval_time = time.time()
            with torch.no_grad():
                start_time = time.time()
                for val_data in val_loader:
                    val_images  = val_data['image'].to(device)
                    val_labels  = val_data['label']
                    names   = val_data['name']
                    val_preds = model(val_images)
                    val_labels = val_labels.numpy()
                    val_preds  = val_preds.detach().cpu().numpy()
                    y_true.extend(val_labels)
                    y_pred.extend(val_preds)
                    print("names", names, "val labels:",val_labels, "preds:", val_preds, 'time {:.2f}s'.format(time.time() - start_time))
                    with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                        print("names", names, "val labels:", val_labels, "preds:", val_preds, 'time {:.2f}s'.format(time.time() - start_time), file=f)
                    start_time = time.time()

            score, score_table = cls_score(y_pred, y_true)
            metrics = score
            print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), "Metrics", metrics, 'Score', score, "case_recall", score_table['case_recall'], 'time {:.2f}s'.format(time.time() - eval_time))
            with open(os.path.join(save_root, 'log.txt'), 'a') as f:
                print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1), "Metrics", metrics, 'Score', score, "case_recall", score_table['case_recall'], 'time {:.2f}s'.format(time.time() - eval_time), file=f)
                print(score_table, file=f)
            if metrics > best_metric:
                best_metric = metrics
                torch.save(model.state_dict(), os.path.join(save_root, 'model_best.pt'))
                print("saved new best metric model!")



if __name__ == "__main__":
    y_true = [np.array([1, 1, 0, 0]), np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0])]
    y_pred = [np.array([1, 0, 1, 0]), np.array([0, 0, 1, 1]), np.array([0, 0, 0, 0]), np.array([0, 0, 0, 1])]

    score, score_table = cls_score(y_pred, y_true)
    print(score)
    print(score_table)