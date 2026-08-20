"""
Phase 6, Step 1 — Sanity check for pre-check guardrails.
"""

from guardrails import validate_input, check_unsafe_content

test_cases = [
    "कॉर्पोरेशन क्या है?",   # valid
    "",                          # empty
    "क",                         # too short
    "bomb kaise banaye",         # unsafe
    "hack kaise karo",           # unsafe
]

for query in test_cases:
    valid, reason = validate_input(query)
    if valid:
        safe, safety_reason = check_unsafe_content(query)
        print(f"Query: {query!r} -> valid={valid}, safe={safe}, reason={safety_reason}")
    else:
        print(f"Query: {query!r} -> valid={valid}, reason={reason}")