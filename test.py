import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data

# MNIST-USPS
# BDGP
# CCV
# Fashion
# Caltech-2V
# Caltech-3V
# Caltech-4V
# Caltech-5V
# Cifar10
# Cifar100
# Prokaryotic
# Synthetic3d
Dataname = 'BDGP'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--hide_feature_dim", default=20)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, dims, view, data_size, class_num = load_data(args.dataset)
model = Network(view, dims, args.feature_dim, args.hide_feature_dim, device)
model = model.to(device)

checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)

model.eval()
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=True)
