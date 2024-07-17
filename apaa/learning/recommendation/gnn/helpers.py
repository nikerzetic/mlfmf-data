import torch

import dgl
import torch.nn.functional as torchfn
import torch.nn as nn
import dgl.function as dglfn
import dgl.nn as dglnn

from sklearn.metrics import roc_auc_score


class GraphSAGE(nn.Module):
    """
    ## Parameters
    - in_feats: the number of input features
    - hidden_sizes: list of integeres representing hidden sizes of convolution layers
    - out_feats: the number of output features

    The first layer has shape [in_feats, hidden_sizes[0]]; the last layer has shape [hidden_sizes[-1], out_feats]. 
    If hidden_sizes is an empty list, the only layer usedd has shape [in_feats, out_feats].
        
    """
    def __init__(self, in_feats: int, hidden_sizes: list[int], out_feats: int):
        super(GraphSAGE, self).__init__()
        previous_feats = in_feats
        self.conv = []
        for h_feats in hidden_sizes:
            self.conv.append(dglnn.SAGEConv(previous_feats, h_feats, "mean"))
            previous_feats = h_feats
        self.conv.append(dglnn.SAGEConv(previous_feats, out_feats, "mean"))
        self.conv = nn.ModuleList(self.conv)
        

    def forward(self, g, in_feat):
        h = in_feat
        for layer in self.conv:
            h = layer(g, h)
            h = torchfn.relu(h)
        return h


class DotPredictor(nn.Module):
    def forward(self, g: dgl.DGLGraph, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(dglfn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]
        

class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.W1 = nn.Linear(h_feats * 2, h_feats)
        self.W2 = nn.Linear(h_feats, 1)

    def apply_edges(self, edges):
        """
        Computes a scalar score for each edge of the given graph.

        Parameters
        ----------
        edges :
            Has three members ``src``, ``dst`` and ``data``, each of
            which is a dictionary representing the features of the
            source nodes, the destination nodes, and the edges
            themselves.

        Returns
        -------
        dict
            A dictionary of new edge features.
        """
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.W2(torchfn.relu(self.W1(h))).squeeze(1)}

    def forward(self, g: dgl.DGLGraph, h: torch.Tensor):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
        
def compute_loss(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])])
    return torchfn.binary_cross_entropy_with_logits(scores, labels)

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)
