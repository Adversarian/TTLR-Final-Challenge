"""Prompt used by the vision router to distinguish explanation from similarity."""

from __future__ import annotations

VISION_ROUTER_PROMPT = """
You classify what the user wants to do with an uploaded product image based on
only their text message. Reply with a single lowercase label:

- Return "explanation" when the user is asking what the image shows, what the
  main object or concept is, or otherwise wants a short description of the
  picture.
- Return "similarity" when the user asks for another product like the one in
  the image, recommendations for comparable items, or anything that clearly
  seeks similar products.

If the message is ambiguous, default to "explanation". Respond with just the
label and no punctuation.
""".strip()

__all__ = ["VISION_ROUTER_PROMPT"]
