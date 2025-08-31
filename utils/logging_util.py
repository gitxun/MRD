# logging_utils.py

from tensorboardX import SummaryWriter

def initialize_tensorboard(log_dir=None):
    """
    Initialize TensorBoard SummaryWriter.
    
    :param log_dir: Optional directory to save logs
    :return: SummaryWriter instance
    """
    return SummaryWriter(log_dir=log_dir)

# logging_utils.py

def log_metrics(writer, train_acc, train_fscore, test_acc, test_fscore, epoch):
    """
    Log training and testing metrics to TensorBoard.

    :param writer: TensorBoard SummaryWriter
    :param train_acc: Training accuracy
    :param train_fscore: Training F1 score
    :param test_acc: Testing accuracy
    :param test_fscore: Testing F1 score
    :param epoch: Current epoch number
    """
    writer.add_scalar('train: accuracy', train_acc, epoch)
    writer.add_scalar('train: fscore', train_fscore, epoch)
    writer.add_scalar('test: accuracy', test_acc, epoch)
    writer.add_scalar('test: fscore', test_fscore, epoch)
