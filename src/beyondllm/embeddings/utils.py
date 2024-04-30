"""Common utils for embeddings."""
import re
import uuid

from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
import os
from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from llama_index.core.bridge.langchain import Embeddings as LCEmbeddings
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.callbacks import CallbackManager
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.utils import get_cache_dir

EmbedType = Union[BaseEmbedding, "LCEmbeddings", str]
from tqdm import tqdm




DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and no prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the \
context information provided."
"""


# generate queries as a convenience function
def generate_qa_embedding_pairs(
    nodes: List[TextNode],
    llm: any,
    qa_generate_prompt_tmpl: str = DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(node_dict.items()):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.predict(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    # construct dataset
    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )


def resolve_embed_model(
    embed_model: Optional[EmbedType] = None,
    callback_manager: Optional[CallbackManager] = None,
) -> BaseEmbedding:
    """Resolve embed model."""
    from llama_index.core.settings import Settings

    try:
        from llama_index.core.bridge.langchain import Embeddings as LCEmbeddings
    except ImportError:
        LCEmbeddings = None  # type: ignore

    if embed_model == "default":
        if os.getenv("IS_TESTING"):
            embed_model = MockEmbedding(embed_dim=8)
            embed_model.callback_manager = callback_manager or Settings.callback_manager
            return embed_model

        try:
            from llama_index.embeddings.openai import (
                OpenAIEmbedding,
            )  # pants: no-infer-dep

            from llama_index.embeddings.openai.utils import (
                validate_openai_api_key,
            )  # pants: no-infer-dep

            embed_model = OpenAIEmbedding()
            validate_openai_api_key(embed_model.api_key)
        except ImportError:
            raise ImportError(
                "`llama-index-embeddings-openai` package not found, "
                "please run `pip install llama-index-embeddings-openai`"
            )
        except ValueError as e:
            raise ValueError(
                "\n******\n"
                "Could not load OpenAI embedding model. "
                "If you intended to use OpenAI, please check your OPENAI_API_KEY.\n"
                "Original error:\n"
                f"{e!s}"
                "\nConsider using embed_model='local'.\n"
                "Visit our documentation for more embedding options: "
                "https://docs.llamaindex.ai/en/stable/module_guides/models/"
                "embeddings.html#modules"
                "\n******"
            )
    # for image multi-modal embeddings
    elif isinstance(embed_model, str) and embed_model.startswith("clip"):
        try:
            from llama_index.embeddings.clip import ClipEmbedding  # pants: no-infer-dep

            clip_model_name = (
                embed_model.split(":")[1] if ":" in embed_model else "ViT-B/32"
            )
            embed_model = ClipEmbedding(model_name=clip_model_name)
        except ImportError as e:
            raise ImportError(
                "`llama-index-embeddings-clip` package not found, "
                "please run `pip install llama-index-embeddings-clip` and `pip install git+https://github.com/openai/CLIP.git`"
            )

    if isinstance(embed_model, str):
        try:
            from llama_index.embeddings.huggingface import (
                HuggingFaceEmbedding,
            )  # pants: no-infer-dep

            splits = embed_model.split(":", 1)
            is_local = splits[0]
            model_name = splits[1] if len(splits) > 1 else None
            if is_local != "local":
                raise ValueError(
                    "embed_model must start with str 'local' or of type BaseEmbedding"
                )

            cache_folder = os.path.join(get_cache_dir(), "models")
            os.makedirs(cache_folder, exist_ok=True)

            embed_model = HuggingFaceEmbedding(
                model_name=model_name, cache_folder=cache_folder
            )
        except ImportError:
            raise ImportError(
                "`llama-index-embeddings-huggingface` package not found, "
                "please run `pip install llama-index-embeddings-huggingface`"
            )

    if LCEmbeddings is not None and isinstance(embed_model, LCEmbeddings):
        try:
            from llama_index.embeddings.langchain import (
                LangchainEmbedding,
            )  # pants: no-infer-dep

            embed_model = LangchainEmbedding(embed_model)
        except ImportError as e:
            raise ImportError(
                "`llama-index-embeddings-langchain` package not found, "
                "please run `pip install llama-index-embeddings-langchain`"
            )

    if embed_model is None:
        print("Embeddings have been explicitly disabled. Using MockEmbedding.")
        embed_model = MockEmbedding(embed_dim=1)

    embed_model.callback_manager = callback_manager or Settings.callback_manager

    return embed_model