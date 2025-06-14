<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>GCN on Cora - Project Report</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      margin: 2rem;
      line-height: 1.6;
      background-color: #fdfdfd;
      color: #333;
    }

    h1, h2, h3 {
      color: #2c3e50;
    }

    .section {
      margin-bottom: 3rem;
    }

    .row {
      display: flex;
      gap: 2rem;
      flex-wrap: wrap;
      justify-content: space-around;
    }

    .image-container {
      flex: 1;
      text-align: center;
    }

    .image-container img {
      width: 100%;
      max-width: 500px;
      border-radius: 8px;
      border: 1px solid #ccc;
    }

    .arxiv-button {
      display: inline-block;
      background-color: #e74c3c;
      color: #fff;
      padding: 0.8rem 1.5rem;
      border-radius: 5px;
      text-decoration: none;
      font-weight: bold;
    }

    blockquote {
      background: #f1f1f1;
      border-left: 5px solid #ccc;
      margin: 1em 0;
      padding: 1em;
      font-style: italic;
    }
  </style>
</head>
<body>

  <div class="section">
    <a class="arxiv-button" href="https://arxiv.org/abs/1609.02907" target="_blank">
      📄 View Original GCN Paper on arXiv
    </a>
  </div>

  <div class="section">
    <h1>Graph Convolutional Networks (GCNs) on Cora Dataset</h1>
    <p>
      This project implements and trains a two-layer Graph Convolutional Network (GCN) for semi-supervised node classification on the Cora citation network dataset. The core idea is to leverage the graph structure and node features to learn expressive embeddings that generalize well to unseen nodes, even with very limited labeled data.
    </p>
  </div>

  <div class="section">
    <h2>Overview and Motivation</h2>
    <p>
      Traditional neural networks don't take into account the relational inductive bias inherent in graph-structured data. GCNs solve this by aggregating features from a node's neighbors during learning, allowing the network to propagate label information through the graph. In the Cora dataset, each node represents a paper, and edges represent citations. The task is to classify each paper into one of several categories using a small set of labeled nodes.
    </p>
    <p>
      This implementation includes a clean architecture, early stopping, loss tracking, and clear visualization of both training metrics and learned embeddings. The model was trained for up to 200 epochs with patience set to 10.
    </p>
  </div>

  <div class="section">
    <h2>Training Results</h2>
    <p>
      Achieved the following performance on the Cora dataset:
    </p>
    <ul>
      <li><strong>Training Accuracy:</strong> 100%</li>
      <li><strong>Validation Accuracy (Best):</strong> 79.80%</li>
      <li><strong>Test Accuracy at Best Val:</strong> 80.80%</li>
    </ul>
  </div>

  <div class="section">
    <h2>Loss & Accuracy Curves</h2>
    <div class="row">
      <div class="image-container">
        <h3>Training Loss</h3>
        <img src="loss.png" alt="Loss Curve">
      </div>
      <div class="image-container">
        <h3>Accuracy</h3>
        <img src="accuracy.png" alt="Accuracy Curve">
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Node Embeddings</h2>
    <p>
      Below, we visualize the learned 2D embeddings using t-SNE. Left shows embeddings from a randomly initialized model (before training). Right shows the same network after training. Clear class separation emerges once the model has been optimized.
    </p>
    <div class="row">
      <div class="image-container">
        <h3>Before Training (Random Weights)</h3>
        <img src="embeddingsInput.png" alt="Random Embeddings">
      </div>
      <div class="image-container">
        <h3>After Training (GCN Embeddings)</h3>
        <img src="embeddingsOutput.png" alt="Trained Embeddings">
      </div>
    </div>
  </div>

  <div class="section">
    <h2>Conclusion</h2>
    <p>
      This project demonstrates that even with a simple two-layer GCN, it's possible to achieve strong performance on the semi-supervised node classification task. The network successfully propagates label information through graph neighborhoods and learns embeddings that are linearly separable. Visualization of embeddings highlights the effectiveness of GCN's message-passing paradigm.
    </p>
  </div>

  <div class="section">
    <h2>Reference</h2>
    <blockquote>
      Kipf, T.N. and Welling, M., 2016. Semi-supervised classification with graph convolutional networks. <i>arXiv preprint arXiv:1609.02907</i>.
    </blockquote>
  </div>

</body>
</html>
