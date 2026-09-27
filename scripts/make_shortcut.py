"""Build a signed iPhone Shortcut that chats with the agent's /chat endpoint.

Run on a Mac (signing uses the `shortcuts` CLI):

    uv run --no-project python scripts/make_shortcut.py -o "Ask Home.shortcut" \\
        [--url URL] [--token TOKEN]

The URL and token are asked for on import, prefilled with what is given here. The text
to send is the shortcut's input: another shortcut passes it with Run Shortcut (after
Dictate Text, say) and gets the reply back as output; run on its own, it asks for text
and shows the reply.
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


def build(url: str, token: str) -> dict:
    url_id, token_id, get_id = (str(uuid.uuid4()).upper() for _ in range(3))
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
            "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
            "WFWorkflowActionParameters": {
                "UUID": get_id,
                "WFURL": text(OBJ, output(url_id, "Text")),
                "WFHTTPMethod": "POST",
                "ShowHeaders": True,
                "WFHTTPHeaders": dictionary(
                    {"Authorization": text(f"Bearer {OBJ}", output(token_id, "Text"))}
                ),
                "WFHTTPBodyType": "JSON",
                "WFJSONValues": dictionary({"text": text(OBJ, SHORTCUT_INPUT)}),
            },
        },
        {
            # hands the reply to a calling shortcut, and shows it when there is none
            "WFWorkflowActionIdentifier": "is.workflow.actions.output",
            "WFWorkflowActionParameters": {
                "WFOutput": text(OBJ, output(get_id, "Contents of URL")),
                "WFNoOutputSurfaceBehavior": "Respond",
            },
        },
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
        ],
        "WFWorkflowInputContentItemClasses": ["WFStringContentItem"],
        "WFWorkflowNoInputBehavior": {
            "Name": "WFWorkflowNoInputBehaviorAskForInput",
            "Parameters": {"ItemClass": "WFStringContentItem"},
        },
        "WFWorkflowOutputContentItemClasses": ["WFStringContentItem"],
        "WFWorkflowHasOutputFallback": True,
        "WFWorkflowTypes": [],
        "WFWorkflowHasShortcutInputVariables": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-o", "--output", default="Ask Home.shortcut")
    parser.add_argument("--url", default="http://192.168.100.121:8952/chat")
    parser.add_argument("--token", default="")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        unsigned = Path(tmp) / "unsigned.shortcut"
        unsigned.write_bytes(plistlib.dumps(build(args.url, args.token)))
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
