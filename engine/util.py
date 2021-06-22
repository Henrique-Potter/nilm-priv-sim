

def gen_fake_users(torch_datasets, data_path, number_of_users, transform):

    import torch

    train_set = torch_datasets(
        data_path, download=True, train=True, transform=transform)
    full_train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True, pin_memory=False)

    # Splitting data set according to the number of users
    user_train_data_size = int(len(train_set) / number_of_users)
    users_train_data = torch.utils.data.random_split(train_set, [user_train_data_size] * number_of_users)

    test_set = torch_datasets(
        data_path, download=True, train=False, transform=transform)
    full_test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, pin_memory=False)

    # Splitting data set according to the number of users
    user_test_data_size = int(len(test_set) / number_of_users)
    users_test_data = torch.utils.data.random_split(test_set, [user_test_data_size] * number_of_users)

    return full_train_loader, full_test_loader, users_test_data, users_train_data


def load_mnist_traindataset(transformes_compose=None, random_gen=None, kwargs=None):
    import torch
    from torchvision import datasets, transforms

    transform = None
    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    if transformes_compose is None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
    else:
        transform = transforms.Compose(transformes_compose)

    # download example data (if necessary) and load it to memory
    trainset = datasets.MNIST('data/MNIST_data/', download=True, train=True, transform=transform)
    if kwargs is not None:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, generator=random_gen, **kwargs)
    else:
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, generator=random_gen,)

    return trainloader


def load_mnist_testdataset(transformes_compose=None, kwargs=None):
    import torch
    from torchvision import datasets, transforms

    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    if transformes_compose is not None:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
    else:
        transform = transforms.Compose(transformes_compose)

    # download example data (if necessary) and load it to memory
    train_set = datasets.MNIST('data/MNIST_data/', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, **kwargs)

    return trainloader


def load_fashion_mnist_traindataset():
    import torch
    from torchvision import datasets, transforms

    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])
                                    ])
    # download example data (if necessary) and load it to memory
    # Download and load training data
    trainset = datasets.FashionMNIST(
        'data/FASHION_MNIST_data/', download=True, train=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, drop_last=True)

    return trainloader


def load_fashion_mnist_testdataset():
    import torch
    from torchvision import datasets, transforms

    # Defines a transform to normalize the data
    # if the img has three channels, you should have three number for mean,
    # for example, img is RGB, mean is [0.5, 0.5, 0.5], the normalize result is R * 0.5, G * 0.5, B * 0.5.
    # If img is grey type that is only one channel, mean should be [0.5], the normalize result is R * 0.5
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])
    # Download and load test data
    testset = datasets.FashionMNIST(
        'data/FASHION_MNIST_data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return testloader


# Function for viewing an image and it's predicted classes.
def view_classification_mnist(img, ps):
    import matplotlib.pyplot as plt
    import numpy as np

    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    plt.show()


def plot_users_convergence(users_tr_data, n_user):

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    x = np.arange(len(users_tr_data[0][0]))

    fig = plt.figure()
    gs = gridspec.GridSpec(2, 3)

    tr_ax = plt.subplot(gs[0, 0])
    tt_ax = plt.subplot(gs[0, 1])
    acc_ax = plt.subplot(gs[0, 2])
    fed_ax = plt.subplot(gs[1, :])

    for i in range(n_user):
        tr_ax.plot(x, users_tr_data[i][0], color='red', linewidth=2)
        tt_ax.plot(x, users_tr_data[i][1], color='green', linewidth=2)
        acc_ax.plot(x, users_tr_data[i][2], color='blue', linewidth=2, linestyle='dashed', label="toto")

    tr_ax.title.set_text('Training loss')
    tt_ax.title.set_text('Test loss')
    acc_ax.title.set_text('Accuracy')

    fed_ax.title.set_text('Fed Model performance')

    plt.legend(('No mask', 'Masked if > 0.5', 'Masked if < -0.5'),
               loc='upper right')
    plt.show()


def plot_losses(loss1, loss2):

    import matplotlib.pyplot as plt

    plt.plot(loss1, label='Training loss')
    plt.plot(loss2, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()


def view_classification_famnist(img, probabilities):

    import matplotlib.pyplot as plt
    import numpy as np

    """Utility to imshow the image and its predicted classes."""
    ps = probabilities.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels([
        'T-shirt/top',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Ankle Boot'
    ], size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

    plt.show()

