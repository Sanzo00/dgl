
from cProfile import label
from turtle import color
import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import RedditDataset
import argparse
import sklearn.metrics

# sampler = dgl.dataloading.NeighborSampler([4, 4])
def get_train_val_dataloader(args, graph, train_nids, valid_nids, device):
  sampler = dgl.dataloading.NeighborSampler(args.sample)
  train_dataloader = dgl.dataloading.DataLoader(
      # The following arguments are specific to DGL's DataLoader.
      graph,              # The graph
      train_nids,         # The node IDs to iterate over in minibatches
      sampler,            # The neighbor sampler
      device=device,      # Put the sampled MFGs on CPU or GPU
      # The following arguments are inherited from PyTorch DataLoader.
      batch_size=args.batch_size,    # Batch size
      shuffle=True,       # Whether to shuffle the nodes for every epoch
      drop_last=False,    # Whether to drop the last incomplete batch
      num_workers=0       # Number of sampler processes
  )
  val_dataloader = dgl.dataloading.DataLoader(
      graph, 
      valid_nids, 
      sampler,
      batch_size=args.batch_size,
      shuffle=False,
      drop_last=False,
      num_workers=0,
      device=device
  )
  return train_dataloader, val_dataloader

# input_nodes, output_nodes, mfgs = example_minibatch = next(iter(train_dataloader))
# print(example_minibatch)
# print(mfgs)
# print("To compute {} nodes' outputs, we need {} nodes' input features".format(len(output_nodes), len(input_nodes))) 


# mfg_0_src = mfgs[0].srcdata[dgl.NID]
# mfg_0_dst = mfgs[0].dstdata[dgl.NID]
# print(mfg_0_src)
# print(mfg_0_dst)
# print(torch.equal(mfg_0_src[:mfgs[0].num_dst_nodes()], mfg_0_dst))

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
  def __init__(self,
                in_feats,
                n_hidden,
                n_classes,
                n_layers,
                activation,
                dropout):
    super(GCN, self).__init__()
    self.layers = nn.ModuleList()
    # input layer
    self.layers.append(GraphConv(in_feats, n_hidden, activation=activation))
    # hidden layers
    for i in range(n_layers - 1):
        self.layers.append(GraphConv(n_hidden, n_hidden, activation=activation))
    # output layer
    self.layers.append(GraphConv(n_hidden, n_classes))
    self.dropout = nn.Dropout(p=dropout)

  def forward(self, mfgs, features):
    # h_dst = features[:mfgs[0].num_dst_nodes()]
    # h = self.layers[0](mfgs[0], (features, h_dst))

    # h = self.dropout(h)
    # h_dst = h[:mfgs[1].num_dst_nodes()]
    # h = self.layers[1](mfgs[1], (h, h_dst))

    h = features
    for i, layer in enumerate(self.layers):
      if i > 0:
        h = self.dropout(h)
      h_dst = h[:mfgs[i].num_dst_nodes()]
      h = self.layers[i](mfgs[i], (h, h_dst))
    return h

# class Model(nn.Module):
#     def __init__(self, in_feats, h_feats, num_classes):
#         super(Model, self).__init__()
#         self.conv1 = SAGEConv(in_feats, h_feats, aggregator_type='mean')
#         self.conv2 = SAGEConv(h_feats, num_classes, aggregator_type='mean')
#         self.h_feats = h_feats

#     def forward(self, mfgs, x):
#         # Lines that are changed are marked with an arrow: "<---"

#         h_dst = x[:mfgs[0].num_dst_nodes()]  # <---
#         h = self.conv1(mfgs[0], (x, h_dst))  # <---
#         h = F.relu(h)
#         h_dst = h[:mfgs[1].num_dst_nodes()]  # <---
#         h = self.conv2(mfgs[1], (h, h_dst))  # <---
        # return h

# model = Model(num_features, 128, num_classes).to(device)
def main(args):
  # load data
  if args.gpu >= 0:
    device = 'cuda'
  else:
    device = 'cpu'
  if args.dataset == 'cora':
    dataset = CoraGraphDataset()
  elif args.dataset == 'citeseer':
    dataset = CiteseerGraphDataset()
  elif args.dataset == 'pubmed':
    dataset = PubmedGraphDataset()
  elif args.dataset == 'reddit':
    dataset = RedditDataset()
  graph = dataset[0]
  # Add reverse edges since ogbn-arxiv is unidirectional.
  # graph = dgl.add_reverse_edges(graph)
  # graph.ndata['label'] = node_labels[:, 0]

  node_features = graph.ndata['feat']
  num_features = node_features.shape[1]
  num_classes = (graph.ndata['label'].max() + 1).item()
  print(graph)
  print('Number of features:', num_features)
  print('Number of classes:', num_classes)

  train_nids = torch.nonzero(graph.ndata['train_mask']).view(-1)
  valid_nids = torch.nonzero(graph.ndata['val_mask']).view(-1)
  test_nids = torch.nonzero(graph.ndata['test_mask']).view(-1)
  print('train_nids.shape:', train_nids.shape)
  print('valid_nids.shape:', valid_nids.shape)
  print('test_nids.shape:', test_nids.shape)
  if args.batch_size == -1:
    args.batch_size = len(train_nids)
  print('batch_size:', args.batch_size)

  model = GCN(num_features, args.n_hidden, num_classes, args.n_layers, F.relu, args.dropout)
  # model = GCN(num_features, args.n_hidden, num_classes, args.n_layers, F.relu, args.dropout).to(device)
  # opt = torch.optim.Adam(model.parameters())
  opt = torch.optim.Adam(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
  train_dataloader, val_dataloader = get_train_val_dataloader(args, graph, train_nids, valid_nids, device)

  # best_accuracy = 0
  # best_model_path = 'model.pt'
  train_acc_list = []
  val_acc_list = []
  for epoch in range(args.n_epochs):
    model.train()
    train_predictions = []
    train_labels = []
    for step, (input_nodes, output_nodes, mfgs) in enumerate(train_dataloader):
      print('epoch {} batch {}'.format(epoch, step))
    # for (input_nodes, output_nodes, mfgs) in train_dataloader:
        # feature copy from CPU to GPU takes place here
      inputs = mfgs[0].srcdata['feat'] # 1644, 1433
      labels = mfgs[-1].dstdata['label'] # 140
      train_labels.append(labels.cpu().numpy())
      predictions = model(mfgs, inputs)
      train_predictions.append(predictions.argmax(1).cpu().numpy())
      loss = F.cross_entropy(predictions, labels)
      opt.zero_grad()
      loss.backward()
      opt.step()

      # accuracy = sklearn.metrics.accuracy_score(labels.cpu().numpy(), predictions.argmax(1).detach().cpu().numpy())
      # print(train_correct, batch_size, train_correct / batch_size)
      # print('step: %d' % step, 'loss: %.03f' % loss.item(), 'acc: %.03f' % accuracy)
      # tq.set_postfix({'loss': '%.03f' % loss.item(), 'acc': '%.03f' % accuracy}, refresh=False)
    train_predictions = np.concatenate(train_predictions)
    train_labels = np.concatenate(train_labels)
    train_accuracy = sklearn.metrics.accuracy_score(train_labels, train_predictions)
    train_acc_list.append(train_accuracy)
    # print('epoch %d' % epoch, 'train_acc %.3f' % train_accuracy)
    model.eval()

    predictions = []
    labels = []
    # val_test_list = []
    with torch.no_grad():
      for input_nodes, output_nodes, mfgs in val_dataloader:
        inputs = mfgs[0].srcdata['feat']
        labels.append(mfgs[-1].dstdata['label'].cpu().numpy())
        predictions.append(model(mfgs, inputs).argmax(1).cpu().numpy())
      predictions = np.concatenate(predictions)
      labels = np.concatenate(labels)
      accuracy = sklearn.metrics.accuracy_score(labels, predictions)
      val_acc_list.append(accuracy)
      # print('Epoch {} Train Accuracy {:.3f} Validation Accuracy {}'.format(epoch, train_accuracy, accuracy))

  # print(val_acc_list)
  # draw(train_acc_list, 'cora 200 epoch train acc (full batch)')
  # draw(val_acc_list, 'cora 200 epoch val acc (full batch)')
  # draw(train_acc_list, 'cora 200 epoch train acc (mini batch)')
  # draw(val_acc_list, 'cora 200 epoch val acc (mini batch)')
  return train_acc_list, val_acc_list

import matplotlib.pyplot as plt
def draw(acc, title=''):
  x = np.arange(len(acc))
  plt.plot(x, acc)
  plt.title(title)
  plt.show()

def draw(y1, y2, desc='', title='', save=None):
  assert len(y1) == len(y2)
  plt.title(title)
  plt.xlabel('Epochs')
  plt.ylabel('Acc')
  x = np.arange(len(y1))
  l1, = plt.plot(x, y1, color='blue')
  l2, = plt.plot(x, y2, color='red')
  print(desc)
  plt.legend(handles=[l1,l2], labels=desc, loc='best')
  # plt.legend(handles=[l1, l2], labels=['a', 'b'], loc='best')
  if save:
    plt.savefig(save)
    plt.close()
  else:
    plt.show()  

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')
  parser.add_argument("--dataset", type=str, default="cora",
                      help="Dataset name ('cora', 'citeseer', 'pubmed').")
  parser.add_argument("--dropout", type=float, default=0.5,
                      help="dropout probability")
  parser.add_argument("--gpu", type=int, default=-1,
                      help="gpu")
  parser.add_argument("--lr", type=float, default=1e-2,
                      help="learning rate")
  parser.add_argument("--n-epochs", type=int, default=200,
                      help="number of training epochs")
  parser.add_argument("--n-hidden", type=int, default=16,
                      help="number of hidden gcn units")
  parser.add_argument("--n-layers", type=int, default=1,
                      help="number of hidden gcn layers")
  parser.add_argument("--batch-size", type=int, default=64,
                      help="batch of each epoch")                        
  parser.add_argument("--weight-decay", type=float, default=5e-4,
                      help="Weight for L2 loss")
  parser.add_argument("--self-loop", action='store_true',
                      help="graph self-loop (default=False)")
  parser.add_argument("--sample", type=str, default="-1, -1",
                      help="sample neighbor number")
  parser.set_defaults(self_loop=False)
  args = parser.parse_args()
  args.sample = [int(item) for item in args.sample.split(',')]
  print(args)

  # full batch vs mini batch
  save_dir = './full-mini-train-test-acc/'
  # for dataset in ['cora', 'citeseer', 'pubmed']:
  for dataset in ['reddit']:
    args.dataset = dataset
    # full batch
    args.sample = [-1, -1]
    args.batch_size = -1
    full_train_acc, full_val_acc = main(args)

    # mini batch
    args.sample = [4, 4]
    args.batch_size = 64
    mini_train_acc, mini_val_acc = main(args)

    draw(full_train_acc, mini_train_acc, desc=['full train acc', 'mini train acc'], title=dataset, save=save_dir+dataset+'-train.png')
    draw(full_val_acc, mini_val_acc, desc=['full val acc', 'mini val acc'], title=dataset, save=save_dir+dataset+'-val.png')
