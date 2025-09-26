import torch

"""

This module consists of a series of utility functions for this project.

"""

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar", additional_info=None):
    """
    Used to save checkpoints of the models when training
    """
    print("=> Saving Checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    if additional_info is not None:
        checkpoint.update(additional_info)
    
    # Try to save the checkpoint
    try:
        torch.save(checkpoint, filename)
        print("=> Checkpoint was successfully saved")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load_checkpoint(filename, model, optimizer, device):
    """
    Used to load a checkpoint for continued training
    """
    print(f"=> Loading checkpoint from '{filename}'")
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if 'additional_info' in checkpoint:
        return checkpoint['additional_info']  # Return additional info if exists

    
    
