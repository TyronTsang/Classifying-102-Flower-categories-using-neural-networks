import torch
from torch import nn, optim
import torchvision
from torchvision import datasets, transforms, models
import argparse

def my_argparse():
    parser = argparse.ArgumentParser(description = 'Flower Classifier Training')

    parser.add_argument('data_dir', default = 'flowers')
    parser.add_argument('--save_dir', help = 'Set directory to save to the checkpoints', default = 'checkpoint.pth')
    parser.add_argument('--learning_rate', help = 'Set the learning rate', default = 0.003)
    parser.add_argument('--hidden_units', help = 'Set the number of hidden units', type= int, default = 3000)
    parser.add_argument('--output_features', help = 'Set the number of output features', type= int, default = 102)
    parser.add_argument('--epochs', help = 'Set the number of Epochs', type= int, default = 3)
    parser.add_argument('--gpu', help = 'Use the GPU for training', default = 'CPU')
    parser.add_argument('--arch', help = 'Choose the architecture', default = 'resnet152')

    return parser.parse_args()

def train_transform(train_dir):
    transform = transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform = transform)
    return train_data

def valid_transform(valid_dir):
    transform = transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])
    validation_transforms = datasets.ImageFolder(valid_dir, transform = transform)
    return validation_transforms

def train_loader(data, batch_size = 64, shuffle=True):
    return torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = shuffle)

def valid_loader(data, batch_size = 64, shuffle=True):
    return torch.utils.data.DataLoader(data, batch_size = batch_size, shuffle = shuffle)

def check_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def load_model(arch):
    exec('model = models.{}(pretrained=True)'.format(arch), globals())

    for param in model.parameters():
        param.requires_grad = False
    return model

def initialize_classifier(model, hidden_units, output_features):
    if hasattr('model', 'classifier'):
        in_features = model.classifier.in_features
    else:
        in_features = model.fc.in_features


    classifier = nn.Sequential(nn.Linear(in_features, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.2),
                               nn.Linear(hidden_units, output_features),
                               nn.LogSoftmax(dim=1))
    return classifier

def train_model(model, trainloader, validloader, optimizer, criterion, epochs=3, print_every=40, step=0):
    for epoch in range(epochs):
        running_loss = 0
        device = check_device()
        for images, labels in trainloader:
            step += 1
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % print_every == 0 :
                test_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)

                        output = model.forward(images)
                        loss = criterion(output, labels)
                        test_loss += loss.item()

                        ps = torch.exp(output)
                        top_p, top_class = ps.top_k(1, dim=1)
                        equals == top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {test_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()

            return model

def save_checkpoint(model, optimizer, class_to_indx, arch, path, hidden_units, output_features):
    model.class_to_indx = class_to_indx
    checkpoint = {'state_dict': model.state_dict(),
                  'hidden_units' : hidden_units,
                  'output_features': output_features,
                  'arch': arch,
                  'class_to_indx': model.class_to_indx}

    torch.save(checkpoint, path)

    return 


def main():
    args = my_argparse()

    data_dir = args.data_dir
    save_path = args.save_dir
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    output_features = args.output_features
    epochs = args.epochs
    gpu = args.gpu
    arch = args.arch

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_data = train_transform(train_dir)
    validation_data = valid_transform(valid_dir)

    trainloader = train_loader(train_data)
    validloader = valid_loader(validation_data)

    if args.gpu:
        device = check_device()

    model = load_model(arch)

    if hasattr('model', 'classifier'):
        model.classifier = initialize_classifier(model, hidden_units, output_features)
    else:
        model.fc = initialize_classifier(model, hidden_units, output_features)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    model.to(device)

    print_every = 40
    steps = 0

    train_model(model, trainloader, validloader, optimizer, criterion, epochs=3, print_every=40, step=0)
    save_checkpoint(model, optimizer, train_data.class_to_idx, save_path, arch, hidden_units, output_features)

if __name__ == '__main__':
    main()
