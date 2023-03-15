from tqdm import tqdm
import numpy as np
import torch


# training
def train1Epoch(epoch_index, model, optimizer, loss_fn, training_loader, model_type):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = []
    if model_type == "VanillaResNet" or model_type == "SegmentedResNet":
        for i, (image, bnpp, _) in tqdm(
            enumerate(training_loader), total=len(training_loader)
        ):
            image, bnpp = (
                image.to(device, dtype=torch.float32, non_blocking=True),
                bnpp.to(device, dtype=torch.float32, non_blocking=True),
            )

            pred = model(image)
            loss = loss_fn(torch.squeeze(pred, 1), bnpp)
            for param in model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    elif model_type == "MultiChannelResNet":
        for i, (image1, image2, image3, bnpp, data, _) in tqdm(
            enumerate(training_loader), total=len(training_loader)
        ):
            image1, image2, image3, bnpp, data = (
                image1.to(device, dtype=torch.float32, non_blocking=True),
                image2.to(device, dtype=torch.float32, non_blocking=True),
                image3.to(device, dtype=torch.float32, non_blocking=True),
                bnpp.to(device, dtype=torch.float32, non_blocking=True),
                data.to(device, dtype=torch.float32, non_blocking=True),
            )

            pred = model(image1, image2, image3, data)
            loss = loss_fn(torch.squeeze(pred, 1), bnpp)
            for param in model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    else:
        for i, (image, bnpp, data, _) in tqdm(
            enumerate(training_loader), total=len(training_loader)
        ):
            image, bnpp, data = (
                image.to(device, dtype=torch.float32, non_blocking=True),
                bnpp.to(device, dtype=torch.float32, non_blocking=True),
                data.to(device, dtype=torch.float32, non_blocking=True),
            )

            pred = model(image, data)
            loss = loss_fn(torch.squeeze(pred, 1), bnpp)
            for param in model.parameters():
                param.grad = None
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    return np.mean(losses)


def test1Epoch(epoch_index, model, loss_fn, valid_loader, model_type):
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = []

    with torch.no_grad():
        if model_type == "VanillaResNet" or model_type == "SegmentedResNet":
            for i, (image, bnpp, _) in tqdm(
                enumerate(valid_loader), total=len(valid_loader)
            ):
                image, bnpp = (
                    image.to(device, dtype=torch.float32, non_blocking=True),
                    bnpp.to(device, dtype=torch.float32, non_blocking=True),
                )
                pred = model(image)
                loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
                losses.append(loss.item())
                image.detach()
                bnpp.detach()
        elif model_type == "MultiChannelResNet":
            for i, (image1, image2, image3, bnpp, data, _) in tqdm(
                enumerate(valid_loader), total=len(valid_loader)
            ):
                image1, image2, image3, bnpp, data = (
                    image1.to(device, dtype=torch.float32, non_blocking=True),
                    image2.to(device, dtype=torch.float32, non_blocking=True),
                    image3.to(device, dtype=torch.float32, non_blocking=True),
                    bnpp.to(device, dtype=torch.float32, non_blocking=True),
                    data.to(device, dtype=torch.float32, non_blocking=True),
                )
                pred = model(image1, image2, image3, data)
                loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
                losses.append(loss.item())
                image1.detach()
                image2.detach()
                image3.detach()
                bnpp.detach()
                data.detach()
        else:
            for i, (image, bnpp, data, _) in tqdm(
                enumerate(valid_loader), total=len(valid_loader)
            ):
                image, bnpp, data = (
                    image.to(device, dtype=torch.float32, non_blocking=True),
                    bnpp.to(device, dtype=torch.float32, non_blocking=True),
                    data.to(device, dtype=torch.float32, non_blocking=True),
                )
                pred = model(image, data)
                loss = loss_fn(torch.squeeze(pred, 1), bnpp).detach()
                losses.append(loss.item())
                image.detach()
                bnpp.detach()
    return np.mean(losses)
