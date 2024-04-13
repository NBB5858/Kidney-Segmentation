import torch
import IPython
import numpy as np


def train_model(model, optimizer, scheduler, loss_fn, max_epochs, train_loader, val_loader=None, save_location=None, load_model_path=None, best_val_loss=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if load_model_path:
        print('loading model')
        checkpoint = torch.load(load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    num_train_batches = len(train_loader)

    if val_loader is not None:
        num_val_batches = len(val_loader)

    model.to(device)

    criterion = loss_fn

    train_losses = []
    val_losses = []

    if best_val_loss is None:
        best_val_loss = 1e9

    out = display(IPython.display.Pretty('Begin training'), display_id=True)
    for epoch in range(max_epochs):

        cum_train_loss = 0.0

        cum_val_loss = 0.0
        avg_val_loss = 0.0

        model.train()

        for batch_idx, (data, target) in enumerate(train_loader):
            out.update(IPython.display.Pretty(f'training on batch {batch_idx+1}/{num_train_batches}'))

            data = data.to(device).to(torch.float32)
            target = target.to(device).unsqueeze(dim=1).to(torch.float32)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):

                output_logits = model(data).unsqueeze(dim=1)

                train_loss = criterion(output_logits, target)

                train_loss.backward()

                optimizer.step()
                scheduler.step()

                cum_train_loss += train_loss.item()

        avg_train_loss = cum_train_loss/num_train_batches
        train_losses.append(avg_train_loss)

        # Model Validation
        if val_loader:

            model.eval()

            for batch_idx, (data, target) in enumerate(val_loader):

                data = data.to(device).to(torch.float32)
                target = target.to(device).unsqueeze(dim=1).to(torch.float32)

                with torch.set_grad_enabled(False):

                    output_logits = model(data).unsqueeze(dim=1)

                    val_loss = criterion(output_logits, target)

                    cum_val_loss += val_loss.item()

            avg_val_loss = cum_val_loss / num_val_batches
            val_losses.append(avg_val_loss)


        last_checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}

        # Save models

        if val_loader:
            if avg_val_loss < best_val_loss:
                best_checkpoint = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
                torch.save(last_checkpoint, save_location)
                best_val_loss = avg_val_loss
                print('saved best model')

        # Print metrics
        print(f'epoch {epoch + 1}/{max_epochs}, train loss: {np.round(avg_train_loss, 8)}, val loss: {np.round(avg_val_loss, 8)}')

    return train_losses, val_losses
