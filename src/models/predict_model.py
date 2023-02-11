import numpy as np
import torch
from tqdm import tqdm

# testing
def test1Epoch(epoch_index, model, loss_fn, valid_loader, tb_writer=None):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = np.array([])

    with torch.no_grad():
        for i, (image, bnpp, _) in tqdm(
            enumerate(valid_loader), total=len(valid_loader)
        ):
            image, bnpp = image.to(device, non_blocking=True), bnpp.to(
                device, non_blocking=True
            )

            pred = model(image)
            loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
            losses = np.append(losses, loss.item())
            image.detach()
            bnpp.detach()
    return np.mean(losses)


def testPretrained(model, valid_loader):
    ### tests pretrained model on validation set
    loss_fn = nn.L1Loss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    tlosses, vlosses = np.array([]), np.array([])
    # loss on unseen test set
    for param in model.parameters():
        param.requires_grad = False
    with torch.no_grad():
        test_loss = test1Epoch(epoch_number, model, loss_fn, test_loader)
    print("test loss:", test_loss)