import numpy as np

def calculate_dice_coefficient(predicted_tensor, ground_truth_tensor):
    # Move tensors to CPU
    predicted_cpu = predicted_tensor.cpu()
    ground_truth_cpu = ground_truth_tensor.cpu()

    # Ensure the input tensors are NumPy arrays
    predicted_np = predicted_cpu.detach().numpy()
    ground_truth_np = ground_truth_cpu.detach().numpy()

    # Flatten the tensors into 1D arrays
    predicted_flat = predicted_np.flatten()
    ground_truth_flat = ground_truth_np.flatten()

    # Calculate True Positive (TP), False Positive (FP), and False Negative (FN)
    TP = np.sum(predicted_flat * ground_truth_flat)
    FP = np.sum(predicted_flat) - TP
    FN = np.sum(ground_truth_flat) - TP

    # Calculate the Dice coefficient
    dice_coefficient = 2 * TP / (2 * TP + FP + FN)
    return dice_coefficient