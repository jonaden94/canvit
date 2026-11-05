"""Class-cued visual search (COCO-Search18 first) on a frozen foveated CanViT, with a
target-conditioned task head and policy. Design: ``claude_dev/search/00-design.md``."""

from .config import SearchAugConfig, SearchConfig, search_policy_config

__all__ = ["SearchAugConfig", "SearchConfig", "search_policy_config"]
