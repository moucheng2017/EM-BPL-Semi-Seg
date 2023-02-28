import torch


def multi_class_label_processing(label,
                                 no_classes):
    '''
    This function turns any multi class label into ensemble of a few binary labels
    Args:
        multi_class_label: 1 x w x h. Label map in the format of integers for classes
    Returns:
    '''
    multi_channel_binary_label = torch.zeros(label.size()[0], no_classes, label.size()[-2], label.size()[-1]).cuda()
    for i in range(no_classes):
        new_channel = torch.zeros_like(label)
        new_channel[label == i] = 1
        multi_channel_binary_label[:, i, :, :] = new_channel
    del new_channel
    del label
    return multi_channel_binary_label.cuda()