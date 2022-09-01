"""Script for mock components."""
import argparse
import logging

import mlflow
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data.dataloader import DataLoader
from torchvision import models, datasets, transforms

logger = logging.getLogger(__name__)

class MnistTrainer():
    def __init__(
        self,
        train_data_dir = './',
        test_data_dir = './',
        model_path=None,
        lr=0.01,
        epochs=1,
        batch_size=64
    ):
        """MNIST Trainer handles train and submit_model tasks. During train_task, it trains a
        simple network on CIFAR10 dataset. For submit_model task, it sends the locally trained model
        (if present) to the server.

        Args:
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            batch_size (int, optional): DataLoader batch size. Defaults to 64.
        """
        
        # Training setup
        self._lr = lr
        self._epochs = epochs
        self.batch_size = batch_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 10)
        self.model.to(self.device)
        self.model_path = model_path
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = SGD(self.model.parameters(), lr=lr, momentum=0.9)
        
        self._train_dataset, self._test_dataset = self.load_dataset(train_data_dir, test_data_dir)
        self._train_loader = DataLoader(self._train_dataset, batch_size=batch_size, shuffle=True)
        self._test_loader = DataLoader(self._test_dataset, batch_size=batch_size, shuffle=True)

    def load_dataset(self,train_data_dir, test_data_dir):
        """Load dataset from {train_data_dir} and {test_data_dir}

        Args:
            train_data_dir(str, optional): Training data directory path
            test_data_dir(str, optional): Testing data directory path
            lr (float, optional): Learning rate. Defaults to 0.01
            epochs (int, optional): Epochs. Defaults to 5
            batch_size (int, optional): DataLoader batch size. Defaults to 64.
        """
        logger.info(f"Train data dir: {train_data_dir}, Test data dir: {test_data_dir}")    
        transformer = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])
        train_dataset = datasets.ImageFolder(train_data_dir, transformer)
        test_dataset = datasets.ImageFolder(test_data_dir, transformer)
        
        return train_dataset, test_dataset     

    def local_train(self, checkpoint):
        """Perform local training for a given number of epochs

        Args:
            checkpoint: Previous model checkpoint from where training has to be started.
        """
        # Basic training
        mlflow.autolog(log_input_examples=True)
        if checkpoint:
            self.model.load_state_dict(torch.load(checkpoint+'/model.pt'))

        mlflow.start_run()
        self.model.train()
        logger.debug("Local training started")  
        for epoch in range(self._epochs):
            running_loss = 0.0
            for i, batch in enumerate(self._train_loader):

                images, labels = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()

                predictions = self.model(images)
                cost = self.loss(predictions, labels)
                cost.backward()
                self.optimizer.step()

                running_loss += cost.cpu().detach().numpy() / images.size()[0]
                if i!=0 and i % 3000 == 0:
                    print( f"Epoch: {epoch}/{self._epochs}, Iteration: {i}, " f"Loss: {running_loss/3000}")
                    running_loss = 0.0
        test_loss, test_acc = self.test()
        logger.info(f"Test Loss: {test_loss} and Test Accuracy: {test_acc}")
        mlflow.end_run()

    def test(self):
        """Test the trained model and report test loss and accuracy

        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self._test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.loss(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self._test_loader.dataset)
        acc = correct / len(self._test_loader.dataset)

        return test_loss, acc

    def execute(self, checkpoint=None):
        """Bundle steps to perform local training, model testing and finally saving the model.
        
        Args:
            checkpoint: Previous model checkpoint from where training has to be started.
        """
        logger.debug("Start training")
        self.local_train(checkpoint)

        logger.debug("Save model")
        torch.save(self.model.state_dict(), self.model_path)
        logger.info(f"Model saved to {self.model_path}")


def get_arg_parser(parser=None):
    """Parse the command line arguments for merge using argparse.

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser):
        an argument parser instance

    Returns:
        ArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the component
    if parser is None:
        parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--train_data", type=str, required=True, help="")
    parser.add_argument("--test_data", type=str, required=True, help="")
    parser.add_argument("--checkpoint", type=str, required=False, help="")
    parser.add_argument("--model", type=str, required=True, help="")
    parser.add_argument("--lr", type=float, required=False, help="Training algorithm's learning rate")
    parser.add_argument("--epochs", type=int, required=False, help="Total number of rounds for local training")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch Size")
    return parser


def run(args):
    """Run script with arguments (the core of the component).

    Args:
        args (argparse.namespace): command line arguments provided to script
    """
    trainer = MnistTrainer(train_data_dir = args.train_data, test_data_dir = args.test_data, model_path=args.model+'/model.pt', lr= args.lr, epochs=args.epochs)
    trainer.execute(args.checkpoint)


def main(cli_args=None):
    """Component main function.

    It parses arguments and executes run() with the right arguments.

    Args:
        cli_args (List[str], optional): list of args to feed script, useful for debugging. Defaults to None.
    """
    # build an arg parser
    parser = get_arg_parser()

    # run the parser on cli args
    args = parser.parse_args(cli_args)

    print(f"Running script with arguments: {args}")
    run(args)


if __name__ == "__main__":
    main()
