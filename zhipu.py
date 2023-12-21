#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2023/12/19 17:36:53
@Author  :   ChenHao
@Contact :   jerrychen1990@gmail.com
'''
from typing import List, Optional
from langchain.llms.base import LLM
import json
import logging
import zhipuai
import numpy as np
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from snippets import batch_process


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatGLM(LLM):

    api_key: str

    model_name: str = "chatglm_turbo"
    """Endpoint URL to use."""
    model_kwargs: Optional[dict] = dict()
    temperature: float = 0.01
    """LLM model temperature from 0 to 10."""
    history: List[List] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    with_history: bool = False
    """Whether to use history or not"""

    @property
    def _llm_type(self) -> str:
        return "chat_glm"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_name": self.model_name},
            **{"model_kwargs": _model_kwargs},
        }

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        zhipuai.api_key = self.api_key
        prompt = self.history + [dict(role="user", content=prompt)]
        response = zhipuai.model_api.invoke(
            model=self.model_name,
            prompt=prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            **self.model_kwargs
        )
        if isinstance(response, dict):
            response = response["data"]["choices"][0]["content"]

        if response.startswith('"') and response.endswith('"'):
            response = response[1:-1]
        response = response.strip()

        if self.with_history:
            self.history.append(dict(role="user", content=prompt))
            self.history.append(dict(role="assistant", content=response))

        # logger.info(f"resp for {prompt}:\n{response}")
        return response


def call_embedding_api(text: str, api_key=None, norm=None, retry_num=2, wait_time=1):

    zhipuai.api_key = api_key
    resp = zhipuai.model_api.invoke(
        model="text_embedding",
        prompt=text
    )
    if resp["code"] != 200:
        logger.error(f"embedding error:{resp['msg']}")
        raise Exception(resp["msg"])
    embedding = resp["data"]["embedding"]
    if norm is not None:
        _norm = 2 if norm == True else norm
        embedding = embedding / np.linalg.norm(embedding, _norm)
    return embedding


class ZhipuEmbedding(Embeddings):
    def __init__(self,  api_key=None, batch_size=16, norm=True):
        self.api_key = api_key
        self.batch_size = batch_size
        self.norm = norm

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        logger.info(f"embedding {len(texts)} with {self.batch_size=}")
        embd_func = batch_process(work_num=self.batch_size, return_list=True)(call_embedding_api)
        embeddings = embd_func(texts, api_key=self.api_key, norm=self.norm)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = call_embedding_api(text, api_key=self.api_key, norm=self.norm)
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        return embedding
