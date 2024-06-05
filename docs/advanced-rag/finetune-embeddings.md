# 📐 Finetune Embeddings

Beyondllm lets you fine-tune embedding models on your own data to achieve more accurate and better results. \
\
You can fine-tune any model available on the [Hugging Face](https://huggingface.co/)&#x20;

### **Step 1 : Importing Modules**

You need an LLM to generate QA pairs for fine-tuning and FineTuneEmbeddings module to fine-tune the model.

```
from beyondllm.llms import GeminiModel
from beyondllm.embeddings import FineTuneEmbeddings

# Initializing llm
llm = llms.GeminiModel()

# calling the finetuning engine
fine_tuned_model = FineTuneEmbeddings()
```

### **Step 2 : Data to FineTune**

You need data to fine-tune your model, It could be 1 or more files so you need to make a list of all the files you want to train your model on.

```
list_of_files = ['your-file-here-1', 'your-file-here-2']
```

### **Step 3 : Training the Model**

Once everything is ready you start training by using the `train` function in FineTuneEmbeddings. &#x20;

**Parameters:**

* **Files :** The list of files you want to train your model on.
* **Model name :** The model you want to fine-tune.&#x20;
* **LLM :** Language model to generate the dataset for fine-tuning.&#x20;
* **Output path :** The path where your embedding model will be saved.&#x20;

```
# Training the embedding model
embed_model = fine_tuned_model.train(list_of_files, "BAAI/bge-small-en-v1.5", llm, "fintune")
```

### **(Optional)  Step 4 : Loading the model**&#x20;

Optionally, If you have already fine-tuned your model and utilize it again, you can do so with the `load_model` function

**Parameters:**

* **Path :** The path where you saved the model after fine-tuning

```
# Option to load an already fine-tuned model
embed_model = fine_tuned_model.load_model("fintune")
```

### **Step 5 : Voila, Use your embedding model**

Setup your retriever using the fine-tuned model and use it in your use case.&#x20;

```
retriever = retrieve.auto_retriever(data, embed_model, type="normal", top_k=4)
```
