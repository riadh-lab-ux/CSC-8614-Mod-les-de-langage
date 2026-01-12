# TP2 — Fine-tuning a language model for text classification

**Riadh Khalfallah** 

## Environnement
- **OS :** Windows 11
- **Python :** 3.10.11
- **Installation / activation :**
  ```bash
  python -m venv tp2
  tp2\Scripts\activate
  pip install -r requirements.txt

**seed = 123**

**Versions librairies**

torch : 2.9.1

tiktoken : 0.12.0

tqdm : 4.67.1

pandas : 2.3.3

matplotlib : 3.10.8

tensorflow : 2.20.0

jupyterlab : 4.5.1

## Preparing the model

### Question 2 : Settings
settings is dictionary. Its structure is a set of GPT-2 configuration hyperparameters with the following keys: ['n_vocab', 'n_ctx', 'n_embd', 'n_head', 'n_layer']. 

In this run, they take the values: n_vocab=50257, n_ctx=1024, n_embd=768, n_head=12, n_layer=12.

### Question 3: Params

params is dictionary and it stores the model weights as arrays. 
The keys are ['blocks', 'b', 'g', 'wpe', 'wte'], where blocks is a list containing per block params, and b, g, wpe, wte are arrays corresponding to global parameters .

### Question 4: Analyse (_init_)

The cfg argument expected by GPTModel is a Python dictionary containing the model hyperparameters used to instantiate the architecture like vocab_size, context_length, emb_dim, n_layers, as well as n_heads and qkv_bias, and drop_rate. 
The downloaded settings object is a dictionary but uses OpenAI-specific key names (n_vocab, n_ctx, n_embd, n_head, n_layer), so it is not directly compatible. 
We therefore create model_config by mapping these keys to the names expected by GPTModel.


## Preparing the data

### Question 5.1

We use df.sample(frac=1, random_state=123) to shuffle the dataset before splitting it into train/test.
This avoids ending up with a biased split. The random_state ensures the shuffle and the split is reproducible, so results can be compared across runs.

### Question 5.2

In the training set, the class distribution is clearly unbalanced:

- Ham: 1726 samples

- Spam: 274 samples 

This means that the majority of the data corresponds to non-spam messages. Such an imbalance can lead the model to favor the majority class during fine-tuning, resulting in high overall accuracy but poor performance on the minority class spam, especially in terms of recall. 
The model may miss many spam messages and to mitigate this issue, one may consider using class-weighted loss functions, resampling techniques, or evaluating with metrics such as precision, recall, and F1-score instead of accuracy alone.

### Question 7 : Number of batches

Train size: 2000
Batch size: 16
Number of batches per epoch: 125

## Fine-tuning

### Question 8.3 :  Why freeze internal layers?

We freeze the internal layers to avoid updating the large pre-trained GPT-2 backbone and only train a small number of parameters so the new classification head, and a small normalization layer.
This makes fine-tuning faster, requires less data, reduces the risk of overfitting, and helps preserve the pre-trained language representations while adapting the model to the spam or ham classification task.

### Question 10 : Training

The training loss decreases from 0.74 to 0.69 over the 4 epochs, showing that the model is effectively learning from the data.
At the same time, the spam accuracy increases strongly, from 28% in epoch 1 to 95% in epoch 4, which indicates that the model progressively learns to recognize the minority class.
This confirms that fine tuning is working and that the class-weighted loss helps the model focus on spam detection. However, the global accuracy slightly decreases at the last epoch, suggesting a trade-off between overall accuracy and minority class performance.

### Question 11 : Impact of changing training params

By training with 1000 samples, 4 epochs, and a learning rate of 1e-4, the model shows a clear improvement in spam detection performance. The spam accuracy on the test set increases from 32.6% to 93.3%, demonstrating that the model adapts well to the spam class despite the strong imbalance. However, the overall test accuracy slightly decreases at the final epoch, which may indicate mild overfitting or a bias introduced by the high class weighting.
This experiment shows that increasing the number of epochs significantly improves minority-class recall, but must be balanced to avoid degrading generalization on the majority class.



