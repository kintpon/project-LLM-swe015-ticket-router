"""Area B - ELIZA-style regex rules for small talk. Used ONLY in the app, before the models."""
import re

RULES = [
    (r"^\s*(hi|hello|hey|good (morning|afternoon|evening))\b[\s!.]*$", "Hello! How can I help you today?"),
    (r"^\s*(thanks|thank you|thx|cheers)\b[\s!.a-z]*$", "You're welcome! Anything else I can do?"),
    (r"^\s*(bye|goodbye|see you|good night)\b[\s!.a-z]*$", "Goodbye! Have a nice day."),
    (r"\b(who|what) are you\b", "I am a ticket router. I send your message to the right team."),
    (r"\bhow are you\b", "I'm just a program, but thanks for asking! What can I do for you?"),
]


def match_rule(text):
    """Return a canned reply if a small-talk rule matches, else None."""
    for pattern, reply in RULES:
        if re.search(pattern, text.lower()):
            return reply
    return None
