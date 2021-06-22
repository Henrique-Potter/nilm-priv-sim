
def yeom_membership_inference(user_model_list, fed_model, tr_data):

    import torch
    import numpy as np

    user_count = 0
    all_users_per_dt_loss = []

    with torch.no_grad():

        for idx in np.arange(len(user_model_list)):

            mod = user_model_list[idx][0]
            optm = user_model_list[idx][1]
            crit = user_model_list[idx][2]
            train_l = user_model_list[idx][3]
            test_l = user_model_list[idx][4]

            user_per_dt_loss = []
            membership = []
            user_total_tr_loss = 0
            user_total_tt_loss = 0

            for images, target_labels in train_l:
                # flatten images into 784 long vector for the input layer
                images = images.view(images.shape[0], -1)
                log_ps = fed_model(images)
                per_instance_loss = crit(log_ps, target_labels, reduction='none')
                user_total_tr_loss += crit(log_ps, target_labels)

                user_per_dt_loss.append(per_instance_loss.cpu().detach().numpy())

            for images, target_labels in test_l:
                # flatten images into 784 long vector for the input layer
                images = images.view(images.shape[0], -1)
                log_ps = fed_model(images)
                per_instance_loss = crit(log_ps, target_labels, reduction='none')
                user_total_tt_loss += crit(log_ps, target_labels)

                user_per_dt_loss.append(per_instance_loss.cpu().detach().numpy())

            user_per_dt_loss = np.vstack(user_per_dt_loss)
            membership.append(np.ones(train_x.shape[0]))

            pred_membership = np.where(user_per_dt_loss <= user_total_tr_loss, 1, 0)

