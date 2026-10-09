"""Shared governance names for the Chat LangChain Lite demo.

One place holds these names. The workspace provisioner
(`scripts/setup_workspace.py`), the chat UI (`web/app.py`), and cleanup all
read them from here, so they cannot drift apart.

Every name carries the presenter suffix. Presenters share one LangSmith
workspace, so an unscoped name would collide.
"""

from __future__ import annotations

from evals.dataset import DEMO_PRESENTER

# LangSmith "Application" grouping. LangSmith reserves the `Application` tag
# key. Every workspace resource for this demo - the tracing project, both
# datasets, the review queue, and the Context Hub repos - carries this tag
# value, so the UI groups them under one application.
APPLICATION_TAG_KEY = "Application"
APPLICATION_NAME = f"chat-langchain-lite-{DEMO_PRESENTER}"

# Feedback keys the chat UI writes. Keep these equal to SCORE_KEY and
# COMMENT_KEY in `web/app.py`.
SCORE_KEY = "user_score"
COMMENT_KEY = "user_comment"

# Annotation queue for thumbs-down traces.
#
# NOTE: keep this name stable. The live queue is already tagged into the
# application and is the target of the "Negative Feedback" rule. A rename
# makes setup_workspace create a SECOND queue while the rule still routes to
# the first, so feedback lands in an untagged queue.
REVIEW_QUEUE_NAME = f"Chat LangChain Lite: Negative Feedback Review - {DEMO_PRESENTER}"
REVIEW_QUEUE_DESCRIPTION = (
    "Traces a user marked thumbs-down (user_score = 0). The rule routes them "
    "here for human review - the 'notify to review' loop."
)

# Run rule that routes any run scored user_score = 0 into the review queue.
# This is the server-side automation. It covers feedback from the chat UI, the
# LangSmith UI, and the SDK alike.
FEEDBACK_RULE_NAME = "Negative Feedback"
FEEDBACK_RULE_FILTER = f'and(eq(feedback_key, "{SCORE_KEY}"), eq(feedback_score, 0))'
