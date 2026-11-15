"""Area B - ELIZA-style regex rules for small talk. Used ONLY in the app, before the models."""
import re

# Every rule must match the WHOLE message (^...$), so "thanks, my oven is still broken"
# is not swallowed by the thanks rule and still reaches the model.
END = r"[\s!?.,]*$"
RULES = [
    (r"^\s*(hi|hello|hey|good (morning|afternoon|evening))( there)?" + END, "Hello! How can I help you today?"),
    (r"^\s*(thanks|thank you|thx|cheers)( a lot| so much| very much)?" + END, "You're welcome! Anything else I can do?"),
    (r"^\s*(bye|goodbye|see you|good night)( bye| for now| later)?" + END, "Goodbye! Have a nice day."),
    (r"^\s*(who|what) are you" + END, "I am a ticket router. I send your message to the right team."),
    (r"^\s*how are you( doing| today)?" + END, "I'm just a program, but thanks for asking! What can I do for you?"),
]


def match_rule(text):
    """Return a canned reply if a small-talk rule matches the whole message, else None."""
    for pattern, reply in RULES:
        if re.search(pattern, text.lower()):
            return reply
    return None
