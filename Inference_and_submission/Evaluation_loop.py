import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import IPython
from Inference_and_submission.Evaluation_dataset import EvaluateSet
from Inference_and_submission.Evaluation_dataset import custom_collator
from Training.Configurations import CFG



def evaluate(model, evaluate_block, perms, thresh=0.5, TTA=False):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()

    output_blocks = []
    used_perms = []
    for perm in perms:
        x = evaluate_block.permute(perm)

        if x.shape[-1] < CFG.image_size or x.shape[-2] < CFG.image_size:
            continue  # skip this perm because we can't tile it.

        reconstructed_masks = np.zeros_like(x, dtype=np.uint8)

        dataset = EvaluateSet(x, perm, tile_size=[CFG.image_size, CFG.image_size], overlap=0.5)

        dataloader = DataLoader(dataset, batch_size=16, collate_fn=custom_collater, shuffle=False, num_workers=2)

        num_batches = len(dataloader)
        out = display(IPython.display.Pretty('Begin evaluation'), display_id=True)

        for batch_idx, (data, data_dicts) in enumerate(dataloader):
            out.update(IPython.display.Pretty(f'evaluation on batch {batch_idx + 1}/{num_batches}; perm {perm}'))

            data = data.to(device)

            with torch.set_grad_enabled(False):

                if TTA == True:
                    output0 = model(data)
                    output1 = torch.rot90(model(torch.rot90(data, 1, dims=[-1, -2])), -1, dims=[-1, -2])
                    output2 = torch.rot90(model(torch.rot90(data, 2, dims=[-1, -2])), -2, dims=[-1, -2])
                    output3 = torch.rot90(model(torch.rot90(data, 3, dims=[-1, -2])), -3, dims=[-1, -2])

                    output4 = torch.flip(model(torch.flip(data, dims=[-1])), dims=[-1])
                    output5 = torch.flip(model(torch.flip(data, dims=[-2])), dims=[-2])

                    # max returns max value and indicies where they occur. don't need the latter, but given for free
                    output = torch.mean(torch.stack([output0, output1, output2, output3, output4, output5], dim=0),
                                        dim=0)
                    print(output.shape)

                else:
                    output = model(data)

                output_probs = nn.Sigmoid()(output).unsqueeze(dim=1)

                preds = (output_probs > thresh)

                ## reassemble tiles
                for pred, data_dict in zip(preds, data_dicts):
                    slice_number = data_dict['slice_number']
                    perm = data_dict['perm']

                    y1, y2 = data_dict['y_range']
                    x1, x2 = data_dict['x_range']

                    pred_tile = pred[0].to('cpu').numpy().astype(np.uint8)

                    reconstructed_masks[slice_number][y1:y2, x1:x2] = np.logical_or(
                        reconstructed_masks[slice_number][y1:y2, x1:x2], pred_tile)

        output_blocks.append(reconstructed_masks)
        used_perms.append(perm)

    return output_blocks, used_perms
