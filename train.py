from tqdm import tqdm
import numpy as np
import torch

# training
def train1Epoch(epoch_index, model, optimizer, loss_fn, training_loader, writer=None):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    for i, (image, bnpp) in tqdm(
        enumerate(training_loader), total=len(training_loader)
    ):
        image, bnpp = image.to(device, non_blocking=True), bnpp.to(
            device, non_blocking=True
        )

        pred = model(image)
        loss = loss_fn(torch.squeeze(pred, 1), bnpp)
        for param in model.parameters():
            param.grad = None
        loss.backward()
        optimizer.step()
        losses = np.append(losses, loss.item())

    return np.mean(losses)


def test1Epoch(epoch_index, model, loss_fn, valid_loader, tb_writer=None):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    with torch.no_grad():
        for i, (image, bnpp) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            image, bnpp = image.to(device, non_blocking=True), bnpp.to(
                device, non_blocking=True
            )
            pred = model(image)
            loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
            losses = np.append(losses, loss.item())
            image.detach()
            bnpp.detach()

    #             out, acts = gcmodel(inpimg)
    #             loss = nn.CrossEntropyLoss()(out,torch.from_numpy(np.array([600])).to(‘cuda:0’))
    #             loss.backward()
    #             grads = gcmodel.get_act_grads().detach().cpu()
    #             pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()
    #             for i in range(acts.shape[1]):
    #              acts[:,i,:,:] += pooled_grads[i]
    #             heatmap_j = torch.mean(acts, dim = 1).squeeze()
    #             heatmap_j_max = heatmap_j.max(axis = 0)[0]
    #             heatmap_j /= heatmap_j_max

    return np.mean(losses)
