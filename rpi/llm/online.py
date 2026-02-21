import asyncio
from typing import AsyncIterator

import httpx
from loguru import logger

from llm.base import BaseLLM


async def check_internet(timeout: float = 3.0) -> bool:
    """Quick internet connectivity check."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.head("https://www.google.com", timeout=timeout)
            return resp.status_code < 500
    except Exception:
        return False
