# LiveKit Agent integrated with Home Assistant MCP Server

This project integrates [LiveKit Agents](https://github.com/livekit/agents) with Home Assistant's [Model Context Protocol (MCP) Server](https://www.home-assistant.io/integrations/mcp_server/), allowing voice control of your smart home through LiveKit agents.

## Overview

This integration enables:
- Voice control of Home Assistant devices through LiveKit Agents
- Real-time audio/video communication with AI assistants
- Access to Home Assistant entities and services via MCP

## Prerequisites

- Home Assistant with MCP Server integration enabled
- LiveKit and OpenAI settings
- Python environment for running the agent

## Setup

1. Configure Home Assistant MCP Server following the [official documentation](https://www.home-assistant.io/integrations/mcp_server/)
2. Set up your LiveKit agent using the [LiveKit Agents framework](https://github.com/livekit/agents)
3. Configure the agent with your Home Assistant URL and access token

## Configuration

```bash
# Example configuration
LIVEKIT_URL = "wss://your-livekit-server.com"
LIVEKIT_API_KEY = "your-api-key"
LIVEKIT_API_SECRET = "your-api-secret"

OPENAI_API_KEY

HOME_ASSISTANT_URL = "http://your-homeassistant:8123/mcp_server/sse"
HOME_ASSISTANT_TOKEN = "your-long-lived-access-token"
```
