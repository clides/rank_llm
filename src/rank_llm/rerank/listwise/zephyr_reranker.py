from typing import Any, List, Optional

from rank_llm.data import Request, Result
from rank_llm.rerank.listwise import RankListwiseOSLLM
from rank_llm.rerank.rankllm import PromptMode


class ZephyrReranker:
    def __init__(
        self,
        model_path: str = "castorini/rank_zephyr_7b_v1_full",
        context_size: int = 4096,
        prompt_mode: Optional[PromptMode] = None,
        prompt_template_path: str = "src/rank_llm/rerank/prompt_templates/rank_zephyr_template.yaml",
        num_few_shot_examples: int = 0,
        few_shot_file: Optional[str] = None,
        device: str = "cuda",
        num_gpus: int = 1,
        variable_passages: bool = True,
        window_size: int = 20,
    ) -> None:
        self._reranker = RankListwiseOSLLM(
            model=model_path,
            context_size=context_size,
            prompt_mode=prompt_mode,
            prompt_template_path=prompt_template_path,
            num_few_shot_examples=num_few_shot_examples,
            few_shot_file=few_shot_file,
            device=device,
            num_gpus=num_gpus,
            variable_passages=variable_passages,
            window_size=window_size,
        )

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> List[Result]:
        """
        Reranks a list of requests using the Zephyr model.

        Args:
            requests (List[Request]): The list of requests. Each request has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                window_size (int): The size of the sliding window for listwise reranking, defualts to 20.
                stride (int): The size of the stride of the sliding window for listwise rernaking, defaults to 10.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            List[Result]: A list containing the reranked results.

        Note:
            check 'reranker.rerank_batch' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=requests,
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )

    def rerank(
        self,
        request: Request,
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        **kwargs: Any,
    ) -> Result:
        """
        Reranks a request using the Zephyr model.

        Args:
            request (Request): The reranking request which has a query and a candidates list.
            rank_start (int, optional): The starting rank for processing. Defaults to 0.
            rank_end (int, optional): The end rank for processing. Defaults to 100.
            shuffle_candidates (bool, optional): Whether to shuffle candidates before reranking. Defaults to False.
            logging (bool, optional): Enables logging of the reranking process. Defaults to False.
            **kwargs: Additional keyword arguments including:
                populate_invocations_history (bool): Whether to populate the history of inference invocations. Defaults to False.
                window_size (int): The size of the sliding window for listwise reranking, defualts to 20.
                stride (int): The size of the stride of the sliding window for listwise rernaking, defaults to 10.
                top_k_retrieve (int): The number of retrieved candidates, when set it is used to cap rank_end and window size.
        Returns:
            Result: the rerank result which contains the reranked candidates.

        Note:
            check 'reranker.rerank_batch' for implementation details of reranking process.
        """
        return self._reranker.rerank_batch(
            requests=[request],
            rank_start=rank_start,
            rank_end=rank_end,
            shuffle_candidates=shuffle_candidates,
            logging=logging,
            **kwargs,
        )
