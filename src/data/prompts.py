"""Prompt templates for synthetic Zulip message generation.

Per D-11: Oversample minority classes (toxic/suicide).
Per D-12: Multi-turn thread prompts for realistic conversations.
Per D-13: Prompt-guided labeling — labels assigned from prompt instruction.
"""
from dataclasses import dataclass
from typing import Literal

LabelType = Literal["toxic", "suicide", "benign"]


@dataclass(frozen=True)
class GenerationPrompt:
    system_prompt: str
    user_prompt: str
    is_suicide: bool
    is_toxicity: bool
    label_type: LabelType


# System prompt for all generations
SYSTEM_BASE = (
    "You are a chat message generator for a content moderation training dataset. "
    "Generate realistic chat messages from a Zulip chat platform. "
    "Each message should be 1-3 sentences. Output ONLY the messages, one per line, "
    "numbered (1., 2., etc.). Do not add any other text."
)


TOXIC_PROMPT = GenerationPrompt(
    system_prompt=SYSTEM_BASE,
    user_prompt=(
        "Generate 10 chat messages that contain toxic content including insults, "
        "threats, obscene language, or identity-based hate. These messages should "
        "be realistic examples of what content moderators would flag. Make them "
        "sound like real chat messages, not generic examples."
    ),
    is_suicide=False,
    is_toxicity=True,
    label_type="toxic",
)


SUICIDE_PROMPT = GenerationPrompt(
    system_prompt=SYSTEM_BASE,
    user_prompt=(
        "Generate 10 chat messages that express suicidal thoughts or self-harm. "
        "These should be realistic examples for training a suicide detection model. "
        "Messages should sound like real distressed chat messages. "
        "IMPORTANT: This is for ML model training only — generating realistic examples "
        "helps save lives by improving detection."
    ),
    is_suicide=True,
    is_toxicity=False,
    label_type="suicide",
)


BENIGN_PROMPT = GenerationPrompt(
    system_prompt=SYSTEM_BASE,
    user_prompt=(
        "Generate 10 normal, friendly chat messages from a Zulip chat platform. "
        "Topics can include: project updates, code reviews, meeting scheduling, "
        "tech discussions, casual conversation, asking for help, sharing links. "
        "Messages should be varied and realistic."
    ),
    is_suicide=False,
    is_toxicity=False,
    label_type="benign",
)


# Label distribution for ~10K rows (D-10, D-11: oversample minorities)
# Real dataset ratio: ~10% toxic, ~22% suicide
# Target synthetic: ~30% toxic, ~30% suicide, ~40% benign (rebalanced)
LABEL_DISTRIBUTION = {
    "toxic": 0.30,    # ~3K rows
    "suicide": 0.30,  # ~3K rows
    "benign": 0.40,   # ~4K rows
}

PROMPTS_BY_LABEL: dict[LabelType, GenerationPrompt] = {
    "toxic": TOXIC_PROMPT,
    "suicide": SUICIDE_PROMPT,
    "benign": BENIGN_PROMPT,
}
