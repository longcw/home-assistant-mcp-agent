"""Text chat with the Home Assistant agent: an A2A endpoint, and a plain-text bridge.

Built on the draft session-persistence API of livekit-agents (livekit/agents#7465).
The bridge at /chat sends each message over A2A to the agent's own endpoint, which
loads the person's conversation (conversations.py). A request names its person by
``user``, an id; none is no one in particular. docs/text-endpoint.md has the API and
the upgrade checklist.
"""
