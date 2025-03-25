# Chapter 3: Coding Attention Mechanisms
it is common to use a deep neural network with two submodules, an encoder and a decoder. The job of the encoder is to first read in and process the entire text, and the decoder then produces the translated text.
Before the advent of transformers, recurrent neural networks (RNNs) were the most popular encoder–decoder architecture for language translation.

**Big limitation of encoder–decoder RNNs** 
The big limitation of encoder–decoder RNNs is that the RNN can’t directly access
earlier hidden states from the encoder during the decoding phase. Consequently, it
relies solely on the current hidden state, which encapsulates all relevant information.

This can lead to a loss of context, especially in complex sentences where dependencies might span long distances.

## 3.2 Capturing data dependencies with attention mechanisms
...researchers found that RNN architectures are not required for building deep neural networks for natural language processing and proposed the original transformer architecture ... including a self-attention mechanism inspired by the Bahdanau attention mechanism.

## 3.3 Attending to different parts of the input with self-attention
> **The “self” in self-attention**
In self-attention, the “self” refers to the mechanism’s ability to compute attention weights by relating different positions within a single input sequence. It assesses and learns the relationships and dependencies between various parts of the input itself, such as words in a sentence or pixels in an image.
This is in contrast to traditional attention mechanisms, where the focus is on the relationships between elements of two different sequences, such as in sequence-to-sequence models where the attention might be between an input sequence and an output sequence,

### 3.3.1 A simple self-attention mechanism without trainable weights
In self-attention, our goal is to calculate context vectors z(i) for each element x(i) in the input sequence. **A context vector can be interpreted as an enriched embedding vector**

...

Context vectors play a crucial role in self-attention. Their purpose is to create enriched representations of each element in an input sequence (like a sentence) by incorporating information from all other elements in the sequence.
This is essential in LLMs, which need to understand the relationship and relevance of words in a sentence to each other. 

The first step of implementing self-attention is to compute the intermediate values ω, referred to as attention scores

In the next step, ..., we normalize each of the attention scores we computed previously. **The main goal behind the normalization is to obtain attention weights that sum up to 1**. This normalization is a convention that is useful for interpretation and maintaining training stability in an LLM ... it’s more common and advisable to use the softmax function for normalization...

In addition, the softmax function ensures that the attention weights are always positive. This makes the output interpretable as probabilities or relative importance, where higher weights indicate greater importance.

the final step, ...: calculating the context vector z(2) by multiplying the embedded input tokens, x(i), with the corresponding attention weights and then summing the resulting vectors.

By convention, the unnormalized attention weights are referred to as **"attention scores"** whereas the normalized attention scores, which sum to 1, are referred to as **"attention weights"**

> **Understanding dot products**
...
Beyond viewing the dot product operation as a mathematical tool that combines
two vectors to yield a scalar value, the dot product is a measure of similarity because it quantifies how closely two vectors are aligned: a higher dot product indicates a greater degree of alignment or similarity between the vectors. In the context of self-attention mechanisms, the dot product determines the extent to which each element in a sequence focuses on, or “attends to,” any other element: **the higher the dot product, the higher the similarity and attention score between two elements**.

## 3.4 Implementing self-attention with trainable weights
Our next step will be to implement the self-attention mechanism used in the original transformer architecture, the GPT models, and most other popular LLMs. This self-attention mechanism is also called **scaled dot-product attention**.

The most notable difference is the introduction of weight matrices that are updated during model training. These trainable weight matrices are crucial so that the model (specifically, the attention module inside the model) can learn to produce “good” context vectors

We will implement the self-attention mechanism step by step by introducing the three trainable weight matrices Wq , Wk , and Wv . These three matrices are used to project the embedded input tokens, x(i) , into query, key, and value vectors, respectively

> **Key, Query and Value**
A **query** is analogous to a search query in a database. It represents the current item (e.g., a word or token in a sentence) the model focuses on or tries to understand. The query is used to probe the other parts of the input sequence to determine how much attention to pay to them.
The **key** is like a database key used for indexing and searching. In the attention mechanism, each item in the input sequence (e.g., each word in a sentence) has an associated key. These keys are used to match the query.
The **value** in this context is similar to the value in a key-value pair in a database. It represents the actual content or representation of the input items. Once the model determines which keys (and thus which parts of the input) are most relevant to the query (the current focus item), it retrieves the corresponding values.

## Computation the attention weights step-by-step
> See the corresponding jupyter notebook for more details

... we initialize the three weight matrices Wq, Wk, and Wv ...
Next, we compute the query, key, and value vectors:

query_2 = x_2 @ W_query

key_2 = x_2 @ W_key

value_2 = x_2 @ W_value

We can obtain all keys and values via matrix multiplication:

keys = inputs @ W_key

values = inputs @ W_value

The second step is to compute the attention scores, ...

Now, we want to go from the attention scores to the attention weights, ...
. We compute the attention weights by scaling the attention scores and  using the softmax function. However, now we scale the attention scores by dividing them by the square root of the embedding dimension of the keys (taking the square root is mathematically the same as exponentiating by 0.5)

Now, the final step is to compute the context vectors, ...

Similar to when we computed the context vector as a weighted sum over the input vectors, we now compute the context vector as a weighted sum over the value vectors.

> **Weight parameters vs. attention weights**
In the weight matrices W, the term “weight” is short for “weight parameters,” the values of a neural network that are optimized during training. This is not to be confused with the attention weights. ... , attention weights determine the extent to which a context vector depends on the different parts of the input (i.e., to what extent the network focuses on different parts of the input).
... weight parameters are the fundamental, learned coefficients that define the network’s connections, while attention weights are dynamic, context-specific values.

> **The rationale behind scaled-dot product attention**
The reason for the normalization by the embedding dimension size is to improve the training performance by avoiding small gradients. For instance, when scaling up the embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large dot products can result in very small gradients during backpropagation due to the softmax function applied to them. As dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero. These small gradients can drastically slow down learning or cause training to stagnate.
The scaling by the square root of the embedding dimension is the reason why this self-attention mechanism is also called scaled-dot product attention.

### 3.4.2
We can improve the SelfAttention_v1 implementation further by utilizing PyTorch’s `nn.Linear` layers, which effectively perform matrix multiplication when the bias units are disabled. Additionally, a significant advantage of using `nn.Linear` instead of manually implementing `nn.Parameter(torch.rand(...))` is that `nn.Linear` has an optimized weight initialization scheme, contributing to more stable and effective model training.

## 3.5 Hiding future words with causal attention
Causal attention, also known as masked attention, is a specialized form of self-attention. It restricts a model to only consider previous and current inputs in a sequence when processing any given token when computing attention scores. This is in contrast to the standard self-attention mechanism, which allows access to the entire input sequence at once.
Now, we will modify the standard self-attention mechanism to create a **causal attention mechanism** ...
We mask out the attention weights above the diagonal, and we normalize the nonmasked attention weights such that the attention weights sum to 1 in each row

>**Information leakage**
When we apply a mask and then renormalize the attention weights, it might initially 
appear that information from future tokens (which we intend to mask) could still influence the current token because their values are part of the softmax calculation. However, **the key insight is that when we renormalize the attention weights after masking,what we’re essentially doing is recalculating the softmax over a smaller subset** (since
masked positions don’t contribute to the softmax value).
The mathematical elegance of softmax is that despite initially including all positions in the denominator, **after masking and renormalizing, the effect of the masked positions is nullified—they don’t contribute to the softmax score in any meaningful way**.
In simpler terms, after masking and renormalization, the distribution of attention weights is as if it was calculated only among the unmasked positions to begin with.
This ensures there’s no information leakage from future (or otherwise masked) tokens as we intended.

### 3.5.2 Masking additional attention weights with dropout
Dropout in deep learning is a technique **where randomly selected hidden layer units are ignored during training, effectively “dropping” them out. This method helps prevent overfitting** by ensuring that a model does not become overly reliant on any specific set of hidden layer units. It’s important to emphasize that **dropout is only used during training and is disabled afterward.**
In the transformer architecture, including models like GPT, dropout in the attention mechanism is typically applied at two specific times: 
1. after calculating the attention weights **or** 
2. after applying the attention weights to the value vectors.

To compensate for the reduction in active elements, the values of the remaining elements in the matrix are scaled up by a factor of 1 / (1 - `dropout_rate`) = 1/0.5 = 2.