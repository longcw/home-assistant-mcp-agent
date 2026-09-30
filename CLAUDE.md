# Project instructions

## Prompts

- `agent/prompt.yaml` is the public prompt and stays general: no rule, example or name that only fits one home, and no language other than the one the rest of it uses.
- Tuning for the user's own home (its areas, devices, scripts, Chinese examples, behaviour they asked for) goes in `agent/prompt.local.yaml`, which the NAS deployment mounts as `AGENT_PROMPT`. Edit the copy in this repo and the one on the NAS (`/vol1/1000/dockers/home-assistant-mcp-agent/agent/prompt.local.yaml`) together; it is bind-mounted, so the next conversation picks it up with no rebuild.
- `agent/*.local.yaml` is git-ignored so it stays visible in the project without being committed. Commit a local prompt (force-add or a `.gitignore` exception) only when it holds no personal info — people's names, addresses, accounts, tokens, IDs — and ask the user first.
