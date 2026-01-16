# TP3 — Parameter-Efficient Fine-Tuning with LoRA

**Riadh Khalfallah** 

## Environnement
- **OS :** Windows 11
- **Python :** 3.10.11
- **Installation / activation :**
  ```bash
  python -m venv tp2
  tp3\Scripts\activate
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

## Setup

**RANDOM_STATE= 123** 

### Question 1
Yes, there is a clear difference. Before applying LoRA, the Transformer blocks contain standard nn.Linear layers.

After applying replace_linear_with_lora , these layers are replaced by LinearWithLoRA, which encapsulates:
- linear: the original frozen nn.Linear layer,
- lora: the added LoRALayer branch.

The presence of LinearWithLoRA in the printed structure of the first Transformer block confirms that LoRA has been correctly injected. The final output head remains a standard nn.Linear layer.

### Question 2

- Trainable parameters: **1,327,104**  
- Total parameters: **164,364,288**  
- Fraction of trainable parameters: **0.81%**

This shows that only a very small fraction of the model parameters are trained, which is consistent with the goal of parameter-efficient fine-tuning using LoRA. Only the low-rank adaptation matrices A and B are updated, while the pretrained GPT weights remain frozen.

### Question 3

Yes, there is a difference compared to the LoRA-only setup.

Previously, with LoRA only, the model had:
- Trainable parameters: 1,327,104  
- Total parameters: 164,364,288  
- Trainable fraction: 0.81%

After adding the new 2-class classification head and making it trainable, we now obtain:
- Trainable parameters: 1,328,642  
- Total parameters: 125,768,450  
- Trainable fraction: 1.06%

The number and fraction of trainable parameters increase because, in addition to the LoRA matrices A and B, the parameters of the new classification head are also trained. This is expected, since the classification head must be learned from scratch for the spam detection task, while LoRA adapts the pretrained GPT representations efficiently with a small number of additional parameters.

### Question 4
The loss shows a clear downward trend during training. It starts very high at the beginning (5.6187) and decreases substantially across batches, reaching small values on  later batches. 
This indicates that the model is learning the spam vs ham classification objective.

In terms of accuracy, the model reaches 80.45% training accuracy after epoch 1 and 96.39% after epoch 2, while the average loss drops from 1.4580 (epoch 1) to 0.1195 (epoch 2). 

This performance is reasonable for the task because spam detection is a relatively simple binary classification problem, the dataset is balanced, and we leverage strong pretrained GPT-2 representations while fine-tuning only a small subset of parameters (LoRA + classification head).


### Question 5
The test set accuracy is 97.32%, which is very close to the training accuracy (96.39%). This indicates good generalization and no significant overfitting.

The high accuracy is reasonable for this binary spam classification task, especially considering that the dataset is balanced and the model benefits from pretrained language representations while only a small subset of parameters is fine-tuned.



