from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Your code here - finish logging the dummy loss and accuracy

    For training, log the training loss every iteration and the average accuracy every epoch
    Call the loss 'train_loss' and accuracy 'train_accuracy'

    For validation, log only the average accuracy every epoch
    Call the accuracy 'val_accuracy'

    Make sure the logging is in the correct spot so the global_step is set correctly,
    for epoch=0, iteration=0: global_step=0
    """
    # strongly simplified training loop
    global_step = 0
    for epoch in range(10):
        train_accuracies = []
        validation_accuracies = []

        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            train_accuracies.extend(dummy_train_accuracy.tolist())
            logger.add_scalar("train_loss", dummy_train_loss, global_step)
            # TODO: log train_loss
            # TODO: save additional metrics to be averaged

            global_step += 1

        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies)
        # TODO: log average train_accuracy
        logger.add_scalar("train_accuracy", avg_train_accuracy, global_step)

        torch.manual_seed(epoch)
        for i in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)
            validation_accuracies.extend(dummy_validation_accuracy.tolist())
            # TODO: save additional metrics to be averaged

        # TODO: log average val_accuracy
        avg_validation_accuracy = sum(validation_accuracies) / len(validation_accuracies)
        logger.add_scalar("val_accuracy", avg_validation_accuracy, global_step)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
