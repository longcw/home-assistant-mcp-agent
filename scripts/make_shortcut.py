"""Build a signed iPhone Shortcut that chats with the agent's /chat endpoint.

Run on a Mac (signing uses the `shortcuts` CLI):

    uv run --no-project python scripts/make_shortcut.py -o "Ask Home.shortcut" \\
        [--url URL] [--token TOKEN] [--user NAME]

The URL, token and the person's name are asked for on import, prefilled with what is
given here. The name keeps each family member's conversation, memories and phone
apart: it matches a person in the card's Settings tab, and left empty the shortcut
talks as no one in particular. It first stops the agent's running turn, so the agent
goes quiet while you speak, then dictates what to send, or takes the shortcut's input
when another shortcut passes one with Run Shortcut. It does not wait for the reply,
which arrives as a Home Assistant notification.
"""

import argparse
import plistlib
import subprocess
import tempfile
import uuid
from pathlib import Path

OBJ = "￼"  # where a variable sits inside a Shortcuts text field
SHORTCUT_INPUT = {"Type": "ExtensionInput"}


def output(action_uuid: str, name: str) -> dict:
    """The variable holding an earlier action's output."""
    return {"OutputUUID": action_uuid, "OutputName": name, "Type": "ActionOutput"}


def text(value: str, *variables: dict) -> dict:
    """A text field: `value`, with each OBJ in it filled by the next variable."""
    positions = [i for i, c in enumerate(value) if c == OBJ]
    attachments = {
        f"{{{pos}, 1}}": variable
        for pos, variable in zip(positions, variables, strict=True)
    }
    return {
        "Value": {"string": value, "attachmentsByRange": attachments},
        "WFSerializationType": "WFTextTokenString",
    }


def dictionary(items: dict[str, dict]) -> dict:
    return {
        "Value": {
            "WFDictionaryFieldValueItems": [
                {"WFItemType": 0, "WFKey": text(key), "WFValue": value}
                for key, value in items.items()
            ]
        },
        "WFSerializationType": "WFDictionaryFieldValue",
    }


def post(url: dict, token_id: str, body: dict[str, dict]) -> dict:
    """A POST of ``body`` as JSON to ``url``, with the agent's bearer token."""
    return {
        "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
        "WFWorkflowActionParameters": {
            "WFURL": url,
            "WFHTTPMethod": "POST",
            "ShowHeaders": True,
            "WFHTTPHeaders": dictionary(
                {"Authorization": text(f"Bearer {OBJ}", output(token_id, "Text"))}
            ),
            "WFHTTPBodyType": "JSON",
            "WFJSONValues": dictionary(body),
        },
    }


def conditional(group: str, mode: int, **params: object) -> dict:
    """One part of an If block: 0 opens it, 1 is Otherwise, 2 ends it."""
    return {
        "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
        "WFWorkflowActionParameters": {
            "GroupingIdentifier": group,
            "WFControlFlowMode": mode,
            **params,
        },
    }


def build(url: str, token: str, user: str) -> dict:
    url_id, token_id, user_id, input_id, dictated_id, said_id, group = (
        str(uuid.uuid4()).upper() for _ in range(7)
    )
    person = {"user": text(OBJ, output(user_id, "Text"))}
    actions = [
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.gettext",
            "WFWorkflowActionParameters": {"UUID": url_id, "WFTextActionText": url},
        },
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.gettext",
            "WFWorkflowActionParameters": {"UUID": token_id, "WFTextActionText": token},
        },
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.gettext",
            "WFWorkflowActionParameters": {"UUID": user_id, "WFTextActionText": user},
        },
        # the agent goes quiet at once, before the person starts speaking
        post(text(f"{OBJ}/cancel", output(url_id, "Text")), token_id, person),
        # text from a calling shortcut, or else what the person says now
        conditional(
            group,
            0,
            WFCondition=100,
            WFInput={
                "Type": "Variable",
                "Variable": {
                    "Value": SHORTCUT_INPUT,
                    "WFSerializationType": "WFTextTokenAttachment",
                },
            },
        ),
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.gettext",
            "WFWorkflowActionParameters": {
                "UUID": input_id,
                "WFTextActionText": text(OBJ, SHORTCUT_INPUT),
            },
        },
        conditional(group, 1),
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.dictatetext",
            "WFWorkflowActionParameters": {"UUID": dictated_id},
        },
        conditional(group, 2, UUID=said_id),
        # the reply is not waited for: it comes back as a notification
        post(
            text(OBJ, output(url_id, "Text")),
            token_id,
            {"text": text(OBJ, output(said_id, "If Result")), **person},
        ),
    ]
    question = {"Category": "Parameter", "ParameterKey": "WFTextActionText"}
    return {
        "WFWorkflowActions": actions,
        "WFWorkflowClientVersion": "2607.0.2",
        "WFWorkflowMinimumClientVersion": 900,
        "WFWorkflowMinimumClientVersionString": "900",
        "WFWorkflowIcon": {
            "WFWorkflowIconStartColor": 463140863,
            "WFWorkflowIconGlyphNumber": 59511,
        },
        "WFWorkflowImportQuestions": [
            {
                **question,
                "ActionIndex": 0,
                "DefaultValue": url,
                "Text": "The agent's /chat URL",
            },
            {
                **question,
                "ActionIndex": 1,
                "DefaultValue": token,
                "Text": "TEXT_API_TOKEN from the server's .env",
            },
            {
                **question,
                "ActionIndex": 2,
                "DefaultValue": user,
                "Text": "Your name, as listed under People in the card's Settings tab",
            },
        ],
        "WFWorkflowInputContentItemClasses": ["WFStringContentItem"],
        "WFWorkflowTypes": [],
        "WFWorkflowHasShortcutInputVariables": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default="Ask Home.shortcut")
    parser.add_argument("--url", required=True, help="the agent's /chat endpoint")
    parser.add_argument("--token", default="")
    parser.add_argument("--user", default="")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        unsigned = Path(tmp) / "unsigned.shortcut"
        unsigned.write_bytes(plistlib.dumps(build(args.url, args.token, args.user)))
        subprocess.run(
            [
                "shortcuts",
                "sign",
                "--mode",
                "anyone",
                "-i",
                unsigned,
                "-o",
                args.output,
            ],
            check=True,
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
