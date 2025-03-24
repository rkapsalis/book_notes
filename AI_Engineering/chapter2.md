# Chapter 2: Understanding Foundation Models
## Modeling
### Model Architecture
There are two problems with seq2seq that Vaswani et al. (2017) addresses.
1. the vanilla seq2seq decoder generates output tokens using only the final hidden state of
the input. Intuitively, this is like generating answers about a book using the book
summary. This limits the quality of the generated outputs. 
2. the RNN encoder and decoder mean that both input processing and output generation are done
sequentially, making it slow for long sequences. If an input is 200 tokens long,
seq2seq has to wait for each input token to finish processing before moving on to the
next

RNNs are especially prone to vanishing and exploding gradients due to their recursive structure. Gradients
must be propagated through many steps, and if they are small, repeated multiplication causes them to shrink
toward zero, making it difficult for the model to learn. Conversely, if the gradients are large, they grow exponentially with each step, leading to instability in the learning process.

The transformer architecture addresses both problems with the attention mechanism. The attention mechanism allows the model to weigh the importance of different input tokens when generating each output token. This is like generating answers
by referencing any page in the book.

Inference for transformer-based language models, therefore, consists of two steps:

**Prefill** <br>
The model processes the input tokens in parallel. This step creates the intermediate state necessary to generate the first output token. This intermediate state
includes the key and value vectors for all input tokens.

**Decode** <br>
The model generates one output token at a time

### Attention mechanism
Under the hood, the attention mechanism leverages key, value,and query vectors:
* **The query vector (Q)** represents the current state of the decoder at each decoding
step. Using the same book summary example, this query vector can be thought of
as the person looking for information to create a summary.
* Each **key vector (K)** represents a previous token. If each previous token is a page
in the book, each key vector is like the page number. Note that at a given decoding step, previous tokens include both input tokens and previously generated
tokens
* Each **value vector (V)** represents the actual value of a previous token, as learned by the model. Each value vector is like the page’s content

The attention mechanism computes how much attention to give an input token by
performing a **dot product** between the query vector and its key vector. A high score means that the model will use more of that page’s content (its value vector) when generating the book’s summary.

**The attention mechanism is almost always multi-headed**. Multiple heads allow the
model to attend to different groups of previous tokens simultaneously. With multi-headed attention, the query, key, and value vectors are split into smaller vectors, each

corresponding to an attention head.

$Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$

## Post-Training
Every model’s post-training is different. However, in general, post-training consists
of two steps:
1. Supervised finetuning (SFT): Finetune the pre-trained model on high-quality
instruction data to optimize models for conversations instead of completion.
2. Preference finetuning: Further finetune the model to output responses that align
with human preference. Preference finetuning is typically done with reinforce‐
ment learning (RL). Techniques for preference finetuning include reinforce‐
ment learning from human feedback (RLHF) (used by GPT-3.5 and Llama 2),
DPO (Direct Preference Optimization) (used by Llama 3), and reinforcement
learning from AI feedback (RLAIF) (potentially used by Claude).