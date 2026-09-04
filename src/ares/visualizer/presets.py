"""Preset Prompts and Benchmark Queries for ARES Visualizer."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PresetPrompt:
    """Standardized preset prompt for interactive demonstration."""
    domain: str
    title: str
    prompt: str
    expected_answer: str
    expected_route: str
    complexity: str  # "Easy", "Medium", "Hard"
    description: str


PRESET_PROMPTS: List[PresetPrompt] = [
    # ─── Math (GSM8K) ────────────────────────────────────────────────────────
    PresetPrompt(
        domain="math",
        title="📐 Apple Store Arithmetic",
        prompt="Solve the following math problem step by step:\nA store sells apples for $2 each and oranges for $3 each. Sarah buys 7 apples and 4 oranges. If she pays with a $50 bill, how much change should she receive?\nAnswer:",
        expected_answer="$24 (Cost = 7*2 + 4*3 = $26; Change = 50 - 26 = $24)",
        expected_route="math",
        complexity="Medium",
        description="Multi-step arithmetic requiring sequential multiplication, addition, and subtraction.",
    ),
    PresetPrompt(
        domain="math",
        title="📐 Speed & Distance Word Problem",
        prompt="Solve the following math problem step by step:\nA train travels 180 km at a speed of 60 km/h. On the return trip, it travels the same distance at 90 km/h. What is the total travel time for the entire round trip?\nAnswer:",
        expected_answer="5 hours (180/60 = 3 hrs; 180/90 = 2 hrs; Total = 3 + 2 = 5 hrs)",
        expected_route="math",
        complexity="Hard",
        description="Multi-stage physics/math rate problem.",
    ),
    # ─── Code (MBPP) ─────────────────────────────────────────────────────────
    PresetPrompt(
        domain="code",
        title="💻 Palindrome Checker Function",
        prompt="Write a Python function to solve the following problem:\nWrite a function is_palindrome(s: str) -> bool that checks if a string is a palindrome, ignoring non-alphanumeric characters and case.\nPython code:\n",
        expected_answer="def is_palindrome(s: str) -> bool:\n    clean = [c.lower() for c in s if c.isalnum()]\n    return clean == clean[::-1]",
        expected_route="code",
        complexity="Easy",
        description="Python string manipulation and two-pointer / slice logic.",
    ),
    PresetPrompt(
        domain="code",
        title="💻 Flatten Nested List",
        prompt="Write a Python function to solve the following problem:\nWrite a function flatten(nested_list: list) -> list that takes an arbitrarily deeply nested list of integers and returns a single flat list.\nPython code:\n",
        expected_answer="def flatten(nested_list):\n    flat = []\n    for item in nested_list:\n        if isinstance(item, list):\n            flat.extend(flatten(item))\n        else:\n            flat.append(item)\n    return flat",
        expected_route="code",
        complexity="Medium",
        description="Recursive list parsing in Python.",
    ),
    # ─── Science (AI2-ARC) ───────────────────────────────────────────────────
    PresetPrompt(
        domain="science",
        title="🧪 Atmospheric Gas Composition",
        prompt="Answer the following science question with the correct option letter:\nQuestion: What is the most abundant gas in Earth's atmosphere?\nChoices:\n(A) Oxygen\n(B) Nitrogen\n(C) Carbon Dioxide\n(D) Argon\nAnswer:",
        expected_answer="(B) Nitrogen (~78% of Earth's atmosphere)",
        expected_route="science",
        complexity="Easy",
        description="Earth science atmospheric chemistry fact retrieval.",
    ),
    PresetPrompt(
        domain="science",
        title="🧪 Energy Transfer in Ecosystems",
        prompt="Answer the following science question with the correct option letter:\nQuestion: In an ecosystem, which organisms directly convert solar energy into chemical energy through photosynthesis?\nChoices:\n(A) Primary consumers\n(B) Decomposers\n(C) Autotrophs\n(D) Apex predators\nAnswer:",
        expected_answer="(C) Autotrophs (Primary producers)",
        expected_route="science",
        complexity="Medium",
        description="Biological energy transfer and ecological classification.",
    ),
    # ─── Reasoning (CommonsenseQA) ──────────────────────────────────────────
    PresetPrompt(
        domain="reasoning",
        title="🧠 Instrumental Sound Observation",
        prompt="Answer the following commonsense reasoning question with the correct option letter:\nQuestion: Where would you expect to hear the sound of an orchestra tuning their instruments before a performance?\nChoices:\n(A) Submarine\n(B) Concert hall\n(C) Grocery store\n(D) Library\n(E) Gas station\nAnswer:",
        expected_answer="(B) Concert hall",
        expected_route="reasoning",
        complexity="Easy",
        description="Commonsense situational spatial association.",
    ),
    PresetPrompt(
        domain="reasoning",
        title="🧠 Temporal Logic & Precedence",
        prompt="Answer the following reasoning question:\nIf Alice arrived at the conference after Bob, and Charlie arrived before Bob, who was the first person to arrive?\nAnswer:",
        expected_answer="Charlie (Charlie arrived before Bob, who arrived before Alice)",
        expected_route="reasoning",
        complexity="Medium",
        description="Transitive temporal relation reasoning.",
    ),
    # ─── General (WikiText) ──────────────────────────────────────────────────
    PresetPrompt(
        domain="general",
        title="📖 Solar System Overview",
        prompt="The Solar System consists of the Sun and the planetary system revolving around it, either directly or indirectly. Of the objects that orbit the Sun directly, the largest are the",
        expected_answer="eight planets, with the remainder being smaller objects, the dwarf planets and small Solar System bodies.",
        expected_route="general",
        complexity="Easy",
        description="Open-ended encyclopedia text completion.",
    ),
]


def get_presets_by_domain(domain: Optional[str] = None) -> List[PresetPrompt]:
    """Filter presets by domain or return all."""
    if domain is None or domain.lower() == "all":
        return PRESET_PROMPTS
    return [p for p in PRESET_PROMPTS if p.domain.lower() == domain.lower()]
