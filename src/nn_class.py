# DS Tools
import numpy as np
import pandas as pd
import os
from glob import glob
import itertools
from collections import Counter
from timeit import default_timer as timer
import tqdm


# Visualizations
import matplotlib.pyplot as plt
import plotly.express as px
plt.rcParams['font.size'] = 14

# Image manipulations
from PIL import Image

# Splitting
from sklearn.model_selection import train_test_split

# Neural Networks
import torch
from torch import Tensor, nn, optim, cuda
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
#from torchsummary import summary
from torch.utils.data import DataLoader, sampler

# Error handling
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class nn_class():

    def __init__(self, batch_size = 64, traindir = f"data/train", validdir = f"data/val", testdir = f"data/test"):
        self.traindir = traindir
        self.validdir = validdir
        self.testdir = testdir
        
        # Change to fit hardware
        self.batch_size = batch_size

        # Whether to train on a gpu
        self.train_on_gpu = cuda.is_available()
        print(f'Train on gpu: {self.train_on_gpu}')

        # Number of gpus
        if self.train_on_gpu:
            self.gpu_count = cuda.device_count()
            print(f'{self.gpu_count} gpus detected.')
            if self.gpu_count > 1:
                self.multi_gpu = True
            else:
                self.multi_gpu = False
        print(self.train_on_gpu, self.multi_gpu)
        
        # Image transformations
        self.image_transforms = {
            # Train uses data augmentation
            'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
        #        transforms.Normalize([0.485, 0.456, 0.406],
        #                             [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
            # Validation does not use augmentation
            'valid':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),

                # Test does not use augmentation
            'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize((0.5,),(0.5,))
                #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        


        # Datasets from folders
        self.data = {
            'train':
            datasets.ImageFolder(root=traindir, transform=self.image_transforms['train']),
            'valid':
            datasets.ImageFolder(root=validdir, transform=self.image_transforms['valid']),
            'test':
            datasets.ImageFolder(root=testdir, transform=self.image_transforms['test'])
        }

        # Dataloader iterators, make sure to shuffle
        self.dataloaders = {
            'train': DataLoader(self.data['train'], batch_size=self.batch_size, shuffle=True,num_workers=10),
            'val': DataLoader(self.data['valid'], batch_size=self.batch_size, shuffle=True,num_workers=10),
            'test': DataLoader(self.data['test'], batch_size=self.batch_size, shuffle=True,num_workers=10)
        }


        # Iterate through the dataloader once
        self.trainiter = iter(self.dataloaders['train'])
        self.features, self.labels = next(self.trainiter)
        self.features.shape, self.labels.shape

        self.categories = []
        for d in os.listdir(traindir):
            self.categories.append(d)

        self.n_classes = len(self.categories)
        print(f'There are {self.n_classes} different classes.')


    def set_model(self, model_name, describe_model=True):
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.save_file_name = f'resnet50-transfer.pt'
            self.checkpoint_path = f'resnet50-transfer.pth'
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.save_file_name = f'resnet18-transfer.pt'
            self.checkpoint_path = f'resnet18-transfer.pth'
        # Doesn't work (yet?)
        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            self.save_file_name = f'vgg16-transfer.pt'
            self.checkpoint_path = f'vgg16-transfer.pth'
        if model_name == 'alexnet':
            self.model = models.alexnet(pretrained=True)
            self.save_file_name = f'alexnet-transfer.pt'
            self.checkpoint_path = f'alexnet-transfer.pth'
        if model_name == 'wide_resnet50_2':
            self.model = models.wide_resnet50_2(pretrained=True)
            self.save_file_name = f'wide_resnet50_2-transfer.pt'
            self.checkpoint_path = f'wide_resnet50_2-transfer.pth'
            
            
            
        # Freeze model weights
        for self.param in self.model.parameters():
            self.param.requires_grad = False

        if describe_model:
            print(self.model)

        self.n_inputs = self.model.fc.in_features

        self.model.fc = nn.Sequential(
                              nn.Linear(self.n_inputs, 256), 
                              nn.ReLU(), 
                              nn.Dropout(0.4),
                              nn.Linear(256, self.n_classes),                   
                              nn.LogSoftmax(dim=1))

        if self.train_on_gpu:
            self.model = self.model.to('cuda')

        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

        #### DO WE NEED THIS?
        self.model.class_to_idx = self.data['train'].class_to_idx
        self.model.idx_to_class = {
            idx: class_
            for class_, idx in self.model.class_to_idx.items()
        }


        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters())
            


    def run_train(self, early_stopping = 3, n_epochs = 30):
        self.model, self.history = self.train(self.model,
                                         self.criterion,
                                         self.optimizer,
                                         self.dataloaders['train'],
                                         self.dataloaders['val'],
                                         save_file_name=self.save_file_name,
                                         max_epochs_stop=early_stopping,
                                         n_epochs=n_epochs,
                                         print_every=1
                                        )
    
    
    def plot_history(self):
        plt.figure(figsize=(8, 6))
        for c in ['train_loss', 'valid_loss']:
            plt.plot(
                self.history[c], label=c)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Average Negative Log Likelihood')
        plt.title('Training and Validation Losses')
        plt.show()
    
    
    def save_checkpoint(self):
        """Save a PyTorch model checkpoint

        Params
        --------
            model (PyTorch model): model to save
            path (str): location to save model. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """

        model_name = self.checkpoint_path.split('-')[0]
        #assert (model_name in ['vgg16', 'resnet50'
        #                       ]), "Path must have the correct model name"

        # Basic details
        checkpoint = {
            'class_to_idx': self.model.class_to_idx,
            'idx_to_class': self.model.idx_to_class,
            'epochs': self.model.epochs,
        }

        # Extract the final classifier and the state dictionary
        if model_name == 'vgg16':
            # Check to see if model was parallelized
            if self.multi_gpu:
                checkpoint['classifier'] = self.model.module.classifier
                checkpoint['state_dict'] = self.model.module.state_dict()
            else:
                checkpoint['classifier'] = self.model.classifier
                checkpoint['state_dict'] = self.model.state_dict()

        elif model_name == 'resnet50':
            if self.multi_gpu:
                checkpoint['fc'] = self.model.module.fc
                checkpoint['state_dict'] = self.model.module.state_dict()
            else:
                checkpoint['fc'] = self.model.fc
                checkpoint['state_dict'] = self.model.state_dict()

        # Add the optimizer
        checkpoint['optimizer'] = self.model.optimizer
        checkpoint['optimizer_state_dict'] = self.model.optimizer.state_dict()

        # Save the data to the path
        torch.save(checkpoint, self.checkpoint_path)
    

    def load_checkpoint(self):
        """Load a PyTorch model checkpoint

        Params
        --------
            path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

        Returns
        --------
            None, save the `model` to `path`

        """

        # Get the model name
        model_name = self.checkpoint_path.split('-')[0]
        #assert (model_name in ['vgg16', 'resnet50'
        #                       ]), "Path must have the correct model name"

        # Load in checkpoint
        checkpoint = torch.load(self.checkpoint_path)

        if model_name == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            # Make sure to set parameters as not trainable
            for self.param in self.model.parameters():
                self.param.requires_grad = False
            self.model.classifier = checkpoint['classifier']

        elif model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Make sure to set parameters as not trainable
            for self.param in self.model.parameters():
                self.param.requires_grad = False
            self.model.fc = checkpoint['fc']

        # Load in the state dict
        self.model.load_state_dict(checkpoint['state_dict'])

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} total gradient parameters.')

        # Move to gpu
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)

        if self.train_on_gpu:
            self.model = self.model.to('cuda')

        # Model basics
        self.model.class_to_idx = checkpoint['class_to_idx']
        self.model.idx_to_class = checkpoint['idx_to_class']
        self.model.epochs = checkpoint['epochs']

        # Optimizer
        self.optimizer = checkpoint['optimizer']
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #return self.model, self.optimizer   
    
    
    def process_image(self,image_path):
        """Process an image path into a PyTorch tensor"""

        image = Image.open(image_path)
        # Resize
        img = image.resize((256, 256))

        # Center crop
        width = 256
        height = 256
        new_width = 224
        new_height = 224

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2
        img = img.crop((left, top, right, bottom))

        # Convert to numpy, transpose color dimension and normalize
        img = np.array(img)
        img = np.repeat(img[..., np.newaxis], 3, -1)
        img = img.transpose((2, 0, 1))
        #img = np.array(img).transpose((2, 0, 1)) / 256
        # Convert to numpy, transpose color dimension and normalize
        img = img / 256

        # Standardization
        #means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
        #stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
        means = np.array([0.5]).reshape((1, 1, 1))
        stds = np.array([0.5]).reshape((1, 1, 1))

        img = img - means
        img = img / stds

        img_tensor = torch.Tensor(img)

        return img_tensor
 


    def predict(self, image_path, topk=5):
        """Make a prediction for an image using a trained model

        Params
        --------
            image_path (str): filename of the image
            model (PyTorch model): trained model for inference
            topk (int): number of top predictions to return

        Returns

        """
        real_class = image_path.split('/')[-2]

        # Convert to pytorch tensor
        img_tensor = self.process_image(image_path)

        # Resize
        if self.train_on_gpu:
            img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
            #img_tensor = img_tensor.view(1, 1, 224, 224).cuda()
        else:
            img_tensor = img_tensor.view(1, 3, 224, 224)
            #img_tensor = img_tensor.view(1, 1, 224, 224)
            #img_tensor = img_tensor.unsqueeze(dim=0)
            #img_tensor = img_tensor#.view()
            #img_tensor = np.repeat(img_tensor[..., np.newaxis], 3, -1)
            #img_tensor = img_tensor.transpose((2,0,1))
            #img = np.array(img).transpose((2, 0, 1)) / 256
            #aaaaaaa=2
            #print(img_tensor)

        # Set to evaluation
        with torch.no_grad():
            self.model.eval()
            # Model outputs log probabilities
            out = self.model(img_tensor)
            ps = torch.exp(out)

            # Find the topk predictions
            topk, topclass = ps.topk(topk, dim=1)

            # Extract the actual classes and probabilities
            top_classes = [
                self.model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
            ]
            top_p = topk.cpu().numpy()[0]

            return img_tensor.cpu().squeeze(), top_p, top_classes, real_class

    def random_test_image(self):
        """Pick a random test image from the test directory"""
        c = np.random.choice(self.categories)
        root = self.testdir +"/"+ c + '/'
        img_path = root + np.random.choice(os.listdir(root))
        return img_path

    
    def display_prediction(self, image_path, topk):
        """Display image and preditions from model"""

        # Get predictions
        img, ps, classes, y_obs = self.predict(image_path, topk)
        # Convert results to dataframe for plotting
        result = pd.DataFrame({'p': ps}, index=classes)

        # Show the image
        plt.figure(figsize=(16, 5))
        ax = plt.subplot(1, 2, 1)
        ax, img = self.imshow_tensor(img, ax=ax)

        # Set title to be the actual class
        ax.set_title(y_obs, size=20)

        ax = plt.subplot(1, 2, 2)
        # Plot a bar plot of predictions
        result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
        plt.xlabel('Predicted Probability')
        plt.tight_layout()
        plt.show()


    def get_categories(self):
        return self.categories;



    def train(self,
              model,
              criterion,
              optimizer,
              train_loader,
              valid_loader,
              save_file_name,
              max_epochs_stop=3,
              n_epochs=20,
              print_every=1):
        """Train a PyTorch Model

        Params
        --------
            model (PyTorch model): cnn to train
            criterion (PyTorch loss): objective to minimize
            optimizer (PyTorch optimizier): optimizer to compute gradients of model parameters
            train_loader (PyTorch dataloader): training dataloader to iterate through
            valid_loader (PyTorch dataloader): validation dataloader used for early stopping
            save_file_name (str ending in '.pt'): file path to save the model state dict
            max_epochs_stop (int): maximum number of epochs with no improvement in validation loss for early stopping
            n_epochs (int): maximum number of training epochs
            print_every (int): frequency of epochs to print training stats

        Returns
        --------
            model (PyTorch model): trained cnn with best weights
            history (DataFrame): history of train and validation loss and accuracy
        """

        # Early stopping intialization
        epochs_no_improve = 0
        valid_loss_min = np.Inf

        valid_max_acc = 0
        history = []

        # Number of epochs already trained (if using loaded in model weights)
        try:
            print(f'Model has been trained for: {self.model.epochs} epochs.\n')
        except:
            self.model.epochs = 0
            print(f'Starting Training from Scratch.\n')

        overall_start = timer()

        # Main loop
        for epoch in range(n_epochs):

            # keep track of training and validation loss each epoch
            train_loss = 0.0
            valid_loss = 0.0

            train_acc = 0
            valid_acc = 0

            # Set to training
            self.model.train()
            start = timer()

            # Training loop
            for ii, (data, target) in enumerate(train_loader):
                # Tensors to gpu
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()

                # Clear gradients
                self.optimizer.zero_grad()
                # Predicted outputs are log probabilities
                output = self.model(data)

                # Loss and backpropagation of gradients
                loss = self.criterion(output, target)
                loss.backward()

                # Update the parameters
                self.optimizer.step()

                # Track train loss by multiplying average loss by number of examples in batch
                train_loss += loss.item() * data.size(0)

                # Calculate accuracy by finding max log probability
                _, pred = torch.max(output, dim=1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                # Need to convert correct tensor from int to float to average
                accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                # Multiply average accuracy times the number of examples in batch
                train_acc += accuracy.item() * data.size(0)

                # Track training progress
                print(
                    f'Epoch: {epoch}\t{100 * (ii + 1) / len(train_loader):.2f}% complete. {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')

            # After training loops ends, start validation
            else:
                self.model.epochs += 1

                # Don't need to keep track of gradients
                with torch.no_grad():
                    # Set to evaluation mode
                    self.model.eval()

                    # Validation loop
                    for data, target in valid_loader:
                        # Tensors to gpu
                        if self.train_on_gpu:
                            data, target = data.cuda(), target.cuda()

                        # Forward pass
                        output = self.model(data)

                        # Validation loss
                        loss = self.criterion(output, target)
                        # Multiply average loss times the number of examples in batch
                        valid_loss += loss.item() * data.size(0)

                        # Calculate validation accuracy
                        _, pred = torch.max(output, dim=1)
                        correct_tensor = pred.eq(target.data.view_as(pred))
                        accuracy = torch.mean(
                            correct_tensor.type(torch.FloatTensor))
                        # Multiply average accuracy times the number of examples
                        valid_acc += accuracy.item() * data.size(0)

                    # Calculate average losses
                    train_loss = train_loss / len(train_loader.dataset)
                    valid_loss = valid_loss / len(valid_loader.dataset)

                    # Calculate average accuracy
                    train_acc = train_acc / len(train_loader.dataset)
                    valid_acc = valid_acc / len(valid_loader.dataset)

                    history.append([train_loss, valid_loss, train_acc, valid_acc])

                    # Print training and validation results
                    if (epoch + 1) % print_every == 0:
                        print(
                            f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \tValidation Loss: {valid_loss:.4f}'
                        )
                        print(
                            f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                        )

                    # Save the model if validation loss decreases
                    if valid_loss < valid_loss_min:
                        # Save model
                        torch.save(self.model.state_dict(), save_file_name)
                        # Track improvement
                        epochs_no_improve = 0
                        valid_loss_min = valid_loss
                        valid_best_acc = valid_acc
                        best_epoch = epoch

                    # Otherwise increment count of epochs with no improvement
                    else:
                        epochs_no_improve += 1
                        # Trigger early stopping
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                            )

                            # Load the best state dict
                            self.model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            self.model.optimizer = self.optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'
                                ])
                            return model, history

        # Attach the optimizer
        self.model.optimizer = self.optimizer
        # Record overall time and print out stats
        total_time = timer() - overall_start
        print(
            f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%'
        )
        print(
            f'{total_time:.2f} total seconds elapsed. {total_time / (epoch):.2f} seconds per epoch.'
        )
        # Format history
        history = pd.DataFrame(
            history,
            columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
        return model, history

    
    def imshow_tensor(self, image, ax=None, title=None):
        """Imshow for Tensor."""

        if ax is None:
            fig, ax = plt.subplots()

        # Set the color channel as the third dimension
        image = image.numpy().transpose((1, 2, 0))

        # Reverse the preprocessing steps
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Clip the image pixel values
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        plt.axis('off')

        return ax, image