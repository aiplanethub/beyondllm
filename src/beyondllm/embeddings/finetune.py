from llama_index.core.node_parser import SimpleNodeParser
from beyondllm.embeddings.utils import generate_qa_embedding_pairs, resolve_embed_model
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core import  SimpleDirectoryReader
from dataclasses import dataclass, field

@dataclass
class FineTuneEmbeddings:
    output_path: str = None
    docs: list = field(init=False, default_factory=list)

    def load_data(self, files):
        reader = SimpleDirectoryReader(input_files=files)
        docs = reader.load_data()
        return docs

    def load_corpus(self, docs, for_training=False, verbose=False):
        parser = SimpleNodeParser.from_defaults()
        split_index = int(len(docs) * 0.7)

        if for_training:
            nodes = parser.get_nodes_from_documents(docs[:split_index], show_progress=verbose)
        else:
            nodes = parser.get_nodes_from_documents(docs[split_index:], show_progress=verbose)

        if verbose:
            print(f'Parsed {len(nodes)} nodes')
        return nodes

    def generate_and_save_datasets(self, docs, llm):
        train_nodes = self.load_corpus(docs, for_training=True, verbose=True)
        val_nodes = self.load_corpus(docs, for_training=False, verbose=True)

        train_dataset = generate_qa_embedding_pairs(train_nodes, llm)
        val_dataset = generate_qa_embedding_pairs(val_nodes, llm)

        train_dataset.save_json("train_dataset.json")
        val_dataset.save_json("val_dataset.json")
        return "train_dataset.json", "val_dataset.json"

    def finetune_model(self, train_file, val_file, model_name):
        try:
            from llama_index.finetuning import SentenceTransformersFinetuneEngine
        except ImportError:
            user_agree = input("The feature you're trying to use require additional packages. Would you like to install it now? [y/N]: ")
            if user_agree.lower() == 'y':
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index-finetuning"])
                subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-index-embeddings-huggingface"])
                
                from llama_index.finetuning import SentenceTransformersFinetuneEngine
            else:
                raise ImportError("The required 'llama-index-finetuning' package is not installed.")
    
        train_dataset = EmbeddingQAFinetuneDataset.from_json(train_file)
        val_dataset = EmbeddingQAFinetuneDataset.from_json(val_file)
        model_output_path = self.output_path or "finetuned_embedding_model"

        finetune_engine = SentenceTransformersFinetuneEngine(
            train_dataset,
            model_id=model_name,
            model_output_path=model_output_path,
            val_dataset=val_dataset
        )
        finetune_engine.finetune()
        return finetune_engine.get_finetuned_model()

    def train(self, files, model_name, llm, output_path=None):
        self.output_path = output_path
        self.docs = self.load_data(files)
        train_file, val_file = self.generate_and_save_datasets(self.docs, llm)
        embed_model = self.finetune_model(train_file, val_file, model_name)
        return embed_model

    def load_model(self, model_path):
        return resolve_embed_model("local:" + model_path)