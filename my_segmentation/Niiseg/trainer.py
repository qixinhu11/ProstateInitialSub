import os, torch
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai import transforms

def dice_score(preds, labels, spe_sen=False):  # on GPU
    ### preds: w,h,d; label: w,h,d
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)

    # print(dice, recall, precision)
    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision


def train(args, train_loader, model, optimizer, loss_function):
    model.train()
    loss_avg = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y = batch["image"].to(args.device), batch["label"].float().to(args.device)  # [B,1,96,96,96]; [B,1,96,96,96]; [B]
        logit_map = model(x)
        loss = loss_function(logit_map, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (DiceCE_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), loss.item())
        )
        loss_avg += loss.item()
        torch.cuda.empty_cache()

    print('Epoch=%d: AVG_DiceCE_loss=%2.5f' % (args.epoch, loss_avg/len(epoch_iterator)))
    return loss_avg/len(epoch_iterator)

def validation(args, test_loader, model):
    model.eval()
    with torch.no_grad():
        post_trans = transforms.Compose([transforms.Activations(sigmoid=True), transforms.AsDiscrete(threshold=0.5)])
        all_dice = 0
        idx = 0
        for val_data in test_loader:
            val_inputs, val_labels = (
                val_data["image"].to(args.device),
                val_data["label"].to(args.device),
            )
            name = val_data['image_meta_dict']['filename_or_obj'][0].split('/')[-2]
            val_outputs = sliding_window_inference(
                inputs=val_inputs,
                roi_size=(args.x, args.y, args.z),
                sw_batch_size=2,
                predictor=model,
                overlap=0.1,
            )
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]

            val_outputs = val_outputs[0][1]
            val_labels  = val_labels[0][0]
            dice, _, _ = dice_score(val_outputs, val_labels)

            with open( os.path.join(args.root_dir, 'log.txt'), 'a') as f:
                print(name, f"Dice: {dice.item()}", file=f)
            all_dice += dice.item()
            idx += 1
        metric = all_dice / idx

    return metric


def trainer(args, scheduler, writer, model, optimizer, loss_function, train_loader, test_loader):
    while args.epoch < args.max_epoch:
        scheduler.step()
        loss_dicece = train(args,train_loader=train_loader, model=model, optimizer=optimizer, loss_function=loss_function)
        writer.add_scalar("train_dicece_loss", loss_dicece, args.epoch)
        writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
        torch.save(checkpoint, os.path.join(args.root_dir, f"last_model.pth"))
        
        if (args.epoch % args.eval_every == 0 and args.epoch != 0): # start to validation
            print("=="*20, "Start validation.")
            with open( os.path.join(args.root_dir, 'log.txt'), 'a') as f:
                print("=="*20, "Start validation.", file=f)
            avg_dice = validation(args,test_loader,model)
            writer.add_scalar("validation_avg_dice", avg_dice, args.epoch)
            checkpoint = {
                    "net": model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    "epoch": args.epoch
                }
            torch.save(checkpoint, os.path.join(args.root_dir, f"model_{args.epoch}.pth"))
        args.epoch += 1