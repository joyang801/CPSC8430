BERT Question Answering on SpokenSQuAD Dataset
This project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for a Question Answering (QA) task using the SpokenSQuAD dataset, a spoken version of the SQuAD dataset with varying levels of Word Error Rates (WER). It aims to fine-tune BERT for QA tasks under different noise conditions to evaluate its robustness.

Prerequisites
Python 3.6+
PyTorch
Transformers library by Hugging Face
Datasets library by Hugging Face
Accelerate library by Hugging Face
evaluate library by Hugging Face
Ensure you have the necessary libraries installed by running:

Dataset Preparation
SpokenSQuAD Dataset: The dataset is a modified version of the SQuAD dataset, tailored for spoken language understanding. It introduces Word Error Rates to simulate real-life spoken scenarios.

Data Preprocessing: A script reformats the original JSON files to create structured entries suitable for the QA task. Each entry contains a question, context, and associated answer(s).

Model and Tokenizer
BERT Model: The bert-base-uncased model is used, leveraging its pre-trained weights for the English language.

Tokenizer: Accompanying the model, the bert-base-uncased tokenizer prepares the textual data for the model by converting words into tokens.

Training and Fine-Tuning
The model is fine-tuned on the preprocessed SpokenSQuAD dataset. The fine-tuning process adjusts the pre-trained model to the specific QA task, enhancing its understanding of spoken language contexts.

The script includes data tokenization, input formatting, and training loops with evaluation checkpoints.

Evaluation
Metrics: The model's performance is assessed using Exact Match (EM) and F1 scores, reflecting the accuracy of the model's answers compared to the ground truth.

Noise Robustness: The model is evaluated under different WER conditions to gauge its resilience to errors inherent in spoken language processing.

Usage
Preprocess the Data: Run the bert.ipynb 

Results
The README should include a section detailing the evaluation results, showcasing the model's performance across different datasets and WER conditions.
