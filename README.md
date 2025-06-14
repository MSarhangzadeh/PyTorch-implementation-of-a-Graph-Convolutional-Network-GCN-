<a href="https://arxiv.org/abs/1609.02907" target="_blank" style="display:inline-block; background:#e74c3c; color:#fff; padding:8px 15px; border-radius:5px; font-weight:bold; text-decoration:none; margin-bottom:20px;">
  📄 View Original GCN Paper on arXiv
</a>

# Graph Convolutional Networks (GCNs) on Cora Dataset

This project implements and trains a two-layer Graph Convolutional Network (GCN) for semi-supervised node classification on the Cora citation network dataset. The core idea is to leverage the graph structure and node features to learn expressive embeddings that generalize well to unseen nodes, even with very limited labeled data.

## Overview and Motivation

Traditional neural networks don't take into account the relational inductive bias inherent in graph-structured data. GCNs solve this by aggregating features from a node's neighbors during learning, allowing the network to propagate label information through the graph. In the Cora dataset, each node represents a paper, and edges represent citations. The task is to classify each paper into one of several categories using a small set of labeled nodes.

This implementation includes a clean architecture, early stopping, loss tracking, and clear visualization of both training metrics and learned embeddings. The model was trained for up to 200 epochs with patience set to 10.

## Training Results

Achieved the following performance on the Cora dataset:

- **Training Accuracy:** 100%
- **Validation Accuracy (Best):** 79.80%
- **Test Accuracy at Best Val:** 80.80%

## Loss & Accuracy Curves

<table>
  <tr>
    <td style="text-align:center;">
      <b>Training Loss</b><br>
      <img src="loss.png" alt="Loss Curve" width="400" style="border:1px solid #ccc; border-radius:8px;">
    </td>
    <td style="text-align:center;">
      <b>Accuracy</b><br>
      <img src="accuracy.png" alt="Accuracy Curve" width="400" style="border:1px solid #ccc; border-radius:8px;">
    </td>
  </tr>
</table>

## Node Embeddings

Below, we visualize the learned 2D embeddings using t-SNE. Left shows embeddings from a randomly initialized model (before training). Right shows the same network after training. Clear class separation emerges once the model has been optimized.

<table>
  <tr>
    <td style="text-align:center;">
      <b>Before Training (Random Weights)</b><br>
      <img src="embeddingsInput.png" alt="Random Embeddings" width="400" style="border:1px solid #ccc; border-radius:8px;">
    </td>
    <td style="text-align:center;">
      <b>After Training (GCN Embeddings)</b><br>
      <img src="embddingsOutput.png" alt="Trained Embeddings" width="400" style="border:1px solid #ccc; border-radius:8px;">
    </td>
  </tr>
</table>

## Conclusion

This project demonstrates that even with a simple two-layer GCN, it's possible to achieve strong performance on the semi-supervised node classification task. The network successfully propagates label information through graph neighborhoods and learns embeddings that are linearly separable. Visualization of embeddings highlights the effectiveness of GCN's message-passing paradigm.

## Reference

> Kipf, T.N. and Welling, M., 2016. Semi-supervised classification with graph convolutional networks. *arXiv preprint arXiv:1609.02907*.
