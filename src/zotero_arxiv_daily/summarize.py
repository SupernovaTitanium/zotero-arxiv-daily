"""TLDR / affiliation generation, kept out of the Paper dataclass.

``generate_tldr_for_paper`` keeps the historical behavior: teaser and full
modes delegate to personal_summary, legacy mode builds a prompt directly; every
failure falls back to the abstract so one bad paper never breaks the email.
"""

import json
import re

import tiktoken
from loguru import logger
from openai import OpenAI

from .llm import request_llm
from .personal_summary import generate_deep_digest, generate_teaser, get_summary_mode


def _truncate(text: str, max_tokens: int) -> str:
    # use gpt-4o tokenizer for estimation
    enc = tiktoken.encoding_for_model("gpt-4o")
    return enc.decode(enc.encode(text)[:max_tokens])


def _generate_tldr_with_llm(paper, openai_client: OpenAI, llm_params: dict) -> str:
    lang = llm_params.get("language", "English")
    prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
    if paper.title:
        prompt += f"Title:\n {paper.title}\n\n"
    if paper.abstract:
        prompt += f"Abstract: {paper.abstract}\n\n"
    if paper.full_text:
        prompt += f"Preview of main content:\n {paper.full_text}\n\n"
    if not paper.full_text and not paper.abstract:
        logger.warning(f"Neither full text nor abstract is provided for {paper.url}")
        return "Failed to generate TLDR. Neither full text nor abstract is provided"
    prompt = _truncate(prompt, 4000)
    return request_llm(
        openai_client,
        llm_params,
        [
            {
                "role": "system",
                "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}.",
            },
            {"role": "user", "content": prompt},
        ],
    )


def generate_tldr_for_paper(paper, openai_client: OpenAI, llm_params: dict) -> str:
    try:
        mode = get_summary_mode(llm_params)
        if mode == "full":
            paper.teaser = generate_teaser(openai_client, llm_params, paper.title, paper.abstract, paper.full_text)
            tldr = generate_deep_digest(openai_client, llm_params, paper.title, paper.abstract, paper.full_text)
            paper.tldr_markdown = tldr
        elif mode == "teaser":
            tldr = generate_teaser(openai_client, llm_params, paper.title, paper.abstract, paper.full_text)
            paper.teaser = tldr
        else:
            tldr = _generate_tldr_with_llm(paper, openai_client, llm_params)
        paper.tldr = tldr
        return tldr
    except Exception as e:
        logger.warning(f"Failed to generate tldr of {paper.url}: {e}")
        paper.tldr = paper.abstract
        return paper.tldr


def _generate_affiliations_with_llm(paper, openai_client: OpenAI, llm_params: dict) -> list[str] | None:
    if paper.full_text is None:
        return None
    prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{paper.full_text}"
    prompt = _truncate(prompt, 2000)
    affiliations = request_llm(
        openai_client,
        llm_params,
        [
            {
                "role": "system",
                "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
    affiliations = json.loads(affiliations)
    affiliations = list(set(affiliations))
    return [str(a) for a in affiliations]


def generate_affiliations_for_paper(paper, openai_client: OpenAI, llm_params: dict) -> list[str] | None:
    try:
        paper.affiliations = _generate_affiliations_with_llm(paper, openai_client, llm_params)
        return paper.affiliations
    except Exception as e:
        logger.warning(f"Failed to generate affiliations of {paper.url}: {e}")
        paper.affiliations = None
        return None
