import net
from torchvision import datasets, transforms
import torch
import sys
import data

def main(mode="test"):
    if(sys.argv[1]=="train"):
        module = net.Mynet()
        train_dataset = data.trainSet("traindata")
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset)
        net.train(10, train_loader, module)
    if(sys.argv[1]=="test"):
        module = torch.load("module/my_model.pkl")
        test_dataset = data.trainSet("traindata")
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
        net.test(test_loader, module)
    if(sys.argv[1]=="print"):
        module = torch.load("module/my_model.pkl")
        print(module.state_dict())

if __name__ == "__main__":
    main()