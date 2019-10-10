import datetime

import torch


def str2bool(s):
    return s.lower() in ('true', '1')


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = datetime.datetime.now()

    def end(self, key="default"):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = datetime.datetime.now() - self.clock[key]
        del self.clock[key]
        return interval.total_seconds()
        

def save_checkpoint(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path, model_path):
    torch.save({
        'epoch': epoch,
        'model': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)
    torch.save(net_state_dict, model_path)
        
        
def load_checkpoint(checkpoint_path):
    return torch.load(checkpoint_path)


def freeze_net_layers(net):
    for param in net.parameters():
        param.requires_grad = False


def store_labels(path, labels):
    with open(path, "w") as f:
        f.write("\n".join(labels))
