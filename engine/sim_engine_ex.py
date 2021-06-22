import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def train_network(epochs, net_model, net_optimizer, net_criterion, net_train_loader, net_test_loader, privatizer=None):

    training_data = {}
    train_losses, test_losses, accuracies, epsilons, alphas = [], [], [], [], []

    if privatizer is not None:
        privatizer.attach(net_optimizer)

    for i in range(epochs):
        running_loss = 0
        for images, target_labels in net_train_loader:
            # flatten images into 784 long vector for the input layer
            images = images.view(images.shape[0], -1)

            # clear gradients because they accumulate
            net_optimizer.zero_grad()
            out = net_model(images)
            batch_loss = net_criterion(out, target_labels)

            # let optimizer update the parameters
            batch_loss.backward()
            net_optimizer.step()

            running_loss += batch_loss.item()
            if privatizer is not None:
                # Less than the inverse of the size of the training dataset.
                delta = 1e-5
                epsilon, curr_alpha = net_optimizer.privacy_engine.get_privacy_spent(delta)
                epsilons.append(epsilon)
                alphas.append(curr_alpha)
                print(
                    f"Train Epoch: {i} \t"
                    f"(ε = {epsilon:.2f}, δ = {delta}) for α = {curr_alpha}"
                )
        else:
            accuracy = 0
            test_loss = 0

            # turn off gradients for validation, saves memory and computation
            with torch.no_grad():
                for images, labels in net_test_loader:
                    images = images.view(images.shape[0], -1)
                    log_ps = net_model(images)
                    test_loss += net_criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    _, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss / len(net_train_loader))
            test_losses.append(test_loss / len(net_test_loader))
            accuracies.append(accuracy / len(net_test_loader))
            print(f'Epoch {i} Accuracy: {accuracy / len(net_test_loader)}')
            print(f'Epoch {i} Training loss: {running_loss / len(net_train_loader)}')
            print(f'Epoch {i} Test loss: {test_loss / len(net_test_loader)}\n\n')

    training_data["train_l"] = train_losses
    training_data["test_l"] = test_losses
    training_data["acc"] = accuracies
    training_data["ep"] = epsilons

    return train_losses, test_losses, accuracies


def gen_user_models(users_train_data, users_test_data, users_split):

    user_models = []
    for i in range(users_split):

        # 2 Layer neural network
        model = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.LogSoftmax(dim=1)
        )

        # Using Adam optimization. Computationally more efficient and performs better with non-stationary objects.
        # Has dynamic learning rate.
        # Good for noisy data.
        temp_optimizer = optim.Adam(model.parameters(), lr=0.001)
        temp_criterion = nn.NLLLoss()

        sub_train_loader = torch.utils.data.DataLoader(users_train_data[i],
                                                       batch_size=64,
                                                       shuffle=True,
                                                       drop_last=True)
        sub_test_loader = torch.utils.data.DataLoader(users_test_data[i],
                                                      batch_size=64,
                                                      shuffle=True)

        user_models.append((model, temp_optimizer, temp_criterion, sub_train_loader, sub_test_loader))

    return user_models


def gaussian_blur(image):
    image = np.array(image)
    image_blur = cv.GaussianBlur(image, (65, 65), 10)
    new_image = image_blur

    return new_image


def simulate_fed_nn(epochs, fed_model, full_test_ldr, number_of_users, user_model_list):

    fed_model_acc = []
    fed_model_test_loss = []
    user_count = 0
    model_training_data = []

    for user_model_data in user_model_list:
        print(f'--- Training User {user_count} model ---')
        mod = user_model_data[0]
        optm = user_model_data[1]
        crit = user_model_data[2]
        train_l = user_model_data[3]
        test_l = user_model_data[4]

        train_losses, test_losses, accuracies = train_network(epochs, mod, optm, crit, train_l, test_l)
        model_training_data.append((train_losses, test_losses, accuracies))

        if user_count is 0:
            for idx, params in enumerate(list(mod.parameters())):
                fed_model_params_list = list(fed_model.parameters())
                fed_model_params_list[idx].data = fed_model_params_list[idx]
        else:
            for idx, params in enumerate(list(mod.parameters())):
                fed_model_params_list = list(fed_model.parameters())
                fed_model_params_list[idx].data = fed_model_params_list[idx].add(params)
            else:
                accuracy = 0
                test_loss = 0

                # turn off gradients for validation, saves memory and computation
                with torch.no_grad():
                    for images, labels in full_test_ldr:
                        images = images.view(images.shape[0], -1)
                        log_ps = fed_model(images)
                        test_loss += crit(log_ps, labels)

                        ps = torch.exp(log_ps)
                        _, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                fed_model_test_loss.append(test_loss / len(full_test_ldr))
                fed_model_acc.append(accuracy / len(full_test_ldr))
                print(f'Accuracy after user {user_count}: {accuracy / len(full_test_ldr)}')
                print(f'Test loss after user {user_count}: {test_loss / len(full_test_ldr)}\n\n')

        user_count += 1

        print(list(fed_model.parameters())[0])

    for idx, params in enumerate(list(fed_model.parameters())):
        params.data = params.data.div(number_of_users)

    print(list(fed_model.parameters())[0])
    return model_training_data


def plot_losses(users_model_train_data, users):
    user_count = 0
    fig, ax = plt.subplots()
    x = np.arange(len(users_model_train_data[0][0]))
    #  model_training_data.append((train_losses, test_losses, accuracies))
    for user_model_data in users_model_train_data:

        ax.plot(user_model_data[0], marker='o', label=f'User {user_count} Training loss')
        ax.plot(user_model_data[1], marker='o', label=f'User {user_count} Validation loss')
        ax.plot(user_model_data[2], marker='o', label=f'User {user_count} Acc')
        user_count += 1

        for i, txt in enumerate(user_model_data[0]):
            ax.annotate(round(txt, 2), (x[i], user_model_data[0][i]))

        for i, txt in enumerate(user_model_data[0]):
            ax.annotate(round(txt, 2), (x[i], user_model_data[1][i]))

        for i, txt in enumerate(user_model_data[0]):
            ax.annotate(round(txt, 2), (x[i], user_model_data[2][i]))

    plt.legend(('No mask', 'Masked if > 0.5', 'Masked if < -0.5'),
               loc='upper right')

    ax.set_ylabel('Scores')
    ax.set_title('Model performance per User')
    ax.legend()

    plt.title('User Performance per epoch')
    plt.show()


def plot_losses2(fed_model_acc, fed_model_test_loss):

    x = np.arange(len(fed_model_acc))
    values = fed_model_acc.extend(fed_model_test_loss)

    fig, ax = plt.subplots()
    plt.bar(x, fed_model_acc)
    plt.bar(x, fed_model_test_loss)

    plt.show()


def plot_losses_bar(users_model_train_data, fed_model_acc, fed_model_test_loss, users):

    user_count = 0
    fake_array1 = [0]*len(users_model_train_data[0][0])
    labels = ['User 0', 'User 1']

    fig, ax = plt.subplots()
    width = 0.35  # the width of the bars
    x = np.arange(len(users_model_train_data))
    # Add some text for labels, title and custom x-axis tick labels, etc.

    #  model_training_data.append((train_losses, test_losses, accuracies))
    for user_model_data in users_model_train_data:

        ax.bar(x, user_model_data[0], color='b', width=width, label=f'Training loss')
        ax.bar(x + width, user_model_data[1], color='r', width=width, label=f'Validation loss')
        ax.bar(x + width*2, user_model_data[2], color='g', width=width, label=f'Accuracy')

        user_count += 1

    ax.set_ylabel('Scores')
    ax.set_title('Model performance per User')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    width = 0.35  # the width of the bars

    fig.tight_layout()
    plt.show()


