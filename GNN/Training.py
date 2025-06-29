import torch.nn.functional as functional
from GNN.Dataset import Dataset
from dgl.nn import EGATConv
import torch.nn as nn
import statistics
import random
import torch
import numpy
import dgl
import os

IAF_root = "../Data/IAF_TrainSet"
#sem = "ST"
#sem = "PR"
sem = "GR"
modelroot = f"models/egat_f23_f1_{sem}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 666
torch.manual_seed(seed)
random.seed(seed)
numpy.random.seed(seed)
dgl.random.seed(seed)


class EGAT(nn.Module):
    def __init__(self,in_node,in_edge,hidden_node,hidden_edge,out_node,out_edge,heads):
        super().__init__()
        self.egat_layers = nn.ModuleList()
        self.egat_layers.append(EGATConv(in_node,in_edge,hidden_node,hidden_edge,heads[0]))
        self.egat_layers.append(EGATConv(hidden_node*heads[0],hidden_edge*heads[0],hidden_node,hidden_edge,heads[1]))
        self.egat_layers.append(EGATConv(hidden_node*heads[1],hidden_edge*heads[1], out_node, out_edge, heads[2]))

    def forward(self, g, node_feats, edge_feats):
        n = node_feats
        e = edge_feats
        for i, layer in enumerate(self.egat_layers):
            new_n, new_e = layer(g, n, e)
            if i != len(self.egat_layers) - 1:
                new_n = functional.relu(new_n)  #fonction d'activation
                n = new_n.flatten(1)  #concaténation
                e = new_e.flatten(1)  #concaténation
            else:
                n = new_n.mean(1)  #moyenne
                e = new_e.mean(1)  #moyenne
        return n, e


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    model = EGAT(23, 1, 6, 6, 4, 1, heads=[5, 3, 3]).to(device)
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters : {total_params}")

    print("Loading Data...")
    iaf_dataset = Dataset(IAF_root, sem, device=device)
    generator = torch.Generator().manual_seed(seed)
    data_loader = dgl.dataloading.GraphDataLoader(iaf_dataset, batch_size=8, shuffle=True)

    print("Start training...")
    model.train()
    for epoch in range(400):
        tot_loss = []
        sum_tot_loss = 0
        batch_count = 0
        for graph in data_loader:
            node_feats = graph.ndata["feat"].to(device)
            edge_feats = graph.edata["is_uncertain"].to(device)
            label = graph.ndata["label"].to(device)
            mask = graph.ndata["mask"].to(device).bool()
            optimizer.zero_grad()  #réinitialisation des gradients avant la rétropropagation
            node_out, edge_out = model(graph, node_feats, edge_feats)
            predicted = (torch.sigmoid(node_out))[mask] #transformation de node_out en valeurs entre 0 et 1
            label = label[mask]
            loss_val = loss(predicted, label)
            loss_val.backward()
            optimizer.step()
            tot_loss.append(loss_val.item())
            sum_tot_loss += loss_val.item()
            batch_count += 1
        if epoch == 8:
            for g in optimizer.param_groups:
                g['lr'] = 0.001
        print("Batchs :", batch_count, "Epoch : ", epoch," Mean : " , statistics.fmean(tot_loss), " Median : ", statistics.median(tot_loss), "Sum loss : ", sum_tot_loss)
    torch.save(model.state_dict(), modelroot)