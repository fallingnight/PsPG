import torch
from torchvision import datasets, transforms
from tqdm import tqdm


def calculate_mean_std(data_dir):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dataset = datasets.ImageFolder(
        data_dir,
        transform=transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        ),
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    rgb_channels = []
    mean = []
    var = []
    num_samples = 0
    for index, (images, labels) in enumerate(tqdm(dataloader, desc="dataset:")):
        images = images.to(device)
        # images = images.transpose((0, 2, 3, 1))  # (batch_size, height, width, channels)
        for i in range(3):  # R,G,B
            rgb_channels.append(images[:, i, :, :].reshape(-1))

        temp_mean = [
            rgb_channels[0].mean(),
            rgb_channels[1].mean(),
            rgb_channels[2].mean(),
        ]
        temp_var = [
            rgb_channels[0].var(unbiased=False),
            rgb_channels[1].var(unbiased=False),
            rgb_channels[2].var(unbiased=False),
        ]
        batch_size = images.shape[0]
        mean.append(temp_mean)
        var.append(temp_var)
        num_samples += batch_size
        rgb_channels = []
        # std.append([np.std(all_pixels_R), np.std(all_pixels_G), np.std(all_pixels_B)])
    print(num_samples)
    mean = torch.tensor(mean)
    var = torch.tensor(var)
    var_all = var.mean(dim=0)
    # std=torch.tensor(std)
    # mean = [m / 255 for m in mean]
    # std = [s / 255 for s in std]
    mean = mean.mean(dim=0)
    std = torch.sqrt(var_all)

    print(f"mean: {mean.tolist()}")
    print(f"std: {std.tolist()}")


calculate_mean_std("chexpert_dataset/train")
