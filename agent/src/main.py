import asyncio
import json
import logging
from typing import Any

from livekit import api, rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    ToolExecutionUpdatedEvent,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.agents.llm import FunctionToolCall
from livekit.agents.llm.utils import execute_function_call

import ha
import scheduler_client as scheduler
from agent import HomeAssistantAgent
from config import (
    MAX_TOOL_OUTPUT_CHARS,
    SESSION_STATE_TOPIC,
    TOOL_CALL_TOPIC,
    settings,
)
from utils import parse_job_metadata, truncate

logger = logging.getLogger("ha-mcp-agent")


# --- Headless scheduled execution ---
# When the scheduler fires a task it dispatches this worker with `kind: "scheduled"` job
# metadata. The entrypoint routes it to run_scheduled_task, which runs headlessly (no
# mic / STT / TTS / user), reports the outcome, notifies Home Assistant, and tears the
# room down.


async def _run_instruction(
    ctx: JobContext, agent: HomeAssistantAgent, text: str
) -> str:
    """Execute a natural-language instruction as a full text-only run (no STT/TTS).

    session.run captures the whole turn — tool calls included — not just one reply.
    """
    session = AgentSession(llm=inference.LLM(settings.llm_model), max_tool_steps=8)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=False, audio_output=False, delete_room_on_close=True
        ),
    )

    async def _finish() -> Any:
        return await session.run(user_input=text)

    try:
        result = await asyncio.wait_for(
            _finish(), timeout=settings.scheduled_run_timeout
        )
    finally:
        await session.aclose()

    replies = [
        ev.item.text_content
        for ev in result.events
        if ev.type == "message" and ev.item.role == "assistant" and ev.item.text_content
    ]
    return replies[-1] if replies else "Done."


async def _run_function_call(agent: HomeAssistantAgent, tool: str, args: dict) -> str:
    """Replay one tool exactly via execute_function_call over the agent's tools."""
    tool_ctx = await agent.tool_context()
    call = FunctionToolCall(name=tool, arguments=json.dumps(args), call_id="scheduled")

    async def _finish() -> Any:
        return await execute_function_call(call, tool_ctx)

    res = await asyncio.wait_for(
        _finish(), timeout=settings.scheduled_run_timeout
    )
    out = res.fnc_call_out
    if out.is_error:
        raise RuntimeError(out.output)
    return out.output


async def run_scheduled_task(ctx: JobContext, meta: dict[str, Any]) -> None:
    task_id = meta.get("task_id", "")
    run_id = meta.get("run_id", "")
    execution = meta.get("execution") or {}
    description = meta.get("description") or "scheduled task"
    logger.info("running scheduled task %s (run %s): %s", task_id, run_id, description)

    await ctx.connect()
    agent = HomeAssistantAgent()
    status = "success"
    result = ""
    try:
        etype = execution.get("type")
        if etype == "function_call":
            args = execution.get("args") or {}
            if isinstance(args, str):
                args = json.loads(args or "{}")
            result = await _run_function_call(agent, execution["tool"], args)
        elif etype == "instruction":
            result = await _run_instruction(ctx, agent, execution.get("text") or "")
        else:
            raise ValueError(f"unknown execution type {etype!r}")
    except Exception as exc:  # noqa: BLE001 - any failure is recorded + notified
        status = "error"
        result = str(exc)
        logger.exception("scheduled task %s failed", task_id)

    result = truncate((result or "").strip(), MAX_TOOL_OUTPUT_CHARS)
    await scheduler.report_run(run_id, status, result)
    if status == "success":
        await ha.notify("Scheduled task done", f"{description}\n\n{result}".strip())
    else:
        message = f"{description}\n\nError: {result}".strip()
        await ha.notify("Scheduled task failed", message)

    try:
        await ctx.delete_room()
    except Exception:
        logger.exception("failed to delete room after scheduled task")


server = AgentServer()


def _forward_tool_events(ctx: JobContext, session: AgentSession) -> None:
    """Stream the tool-execution lifecycle to the frontend over a data channel.

    Each ``tool_execution_updated`` event is serialized and published on
    ``TOOL_CALL_TOPIC``. One consumer preserves ordering, and large outputs are
    clipped to the data-packet size budget (modeled on the upstream async_tool_agent).
    """
    queue: asyncio.Queue[ToolExecutionUpdatedEvent] = asyncio.Queue()

    @session.on("tool_execution_updated")
    def _on_tool_execution_updated(ev: ToolExecutionUpdatedEvent) -> None:
        queue.put_nowait(ev)

    async def _pump() -> None:
        while True:
            ev = await queue.get()
            data = ev.model_dump(mode="json")
            update = data.get("update", {})
            message = update.get("message")
            if isinstance(message, str):
                update["message"] = truncate(message, MAX_TOOL_OUTPUT_CHARS)
            try:
                await ctx.room.local_participant.publish_data(
                    json.dumps(data, ensure_ascii=False),
                    topic=TOOL_CALL_TOPIC,
                    reliable=True,
                )
            except Exception:
                logger.exception("failed to publish tool event")

    task = asyncio.create_task(_pump())

    async def _cancel_pump() -> None:
        task.cancel()

    ctx.add_shutdown_callback(_cancel_pump)


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}

    # A scheduled dispatch (from the scheduler service) runs headlessly and exits: no
    # mic, STT, TTS, or user. Everything below this guard is the interactive path.
    meta = parse_job_metadata(ctx.job.metadata)
    if meta.get("kind") == "scheduled":
        await run_scheduled_task(ctx, meta)
        return

    # Detector reused whenever we switch to auto; STT instance held so we can detach it
    # (stt=None) and rewire the same object later via Agent.update_options. The VAD is
    # the session's bundled Silero default and stays live throughout (local, unbilled).
    turn_detector = inference.TurnDetector()
    stt = inference.STT(settings.stt_model, language=settings.stt_language)

    agent = HomeAssistantAgent()
    session = AgentSession(
        stt=stt,
        llm=inference.LLM(settings.llm_model),
        tts=inference.TTS(
            settings.tts_model, voice=settings.tts_voice, language=settings.tts_language
        ),
        turn_handling=TurnHandlingOptions(turn_detection="manual"),
        max_tool_steps=8,
    )

    await session.start(agent=agent, room=ctx.room)
    await ctx.connect()

    _forward_tool_events(ctx, session)

    # --- Session state the frontend mirrors, and the controls that mutate it. ---
    #
    # Deliberately simple: STT is billed continuously, so it follows the mic (see
    # _set_audio_input) — live while audio input is on, torn down after STT_IDLE_TIMEOUT
    # once gated. TTS (audio output) and text chat are independent, so the agent can run
    # as pure text with zero speech cost. The card drives it all over RPCs; the agent
    # boots dormant + text-only so an idle card costs only its connection. Same in both
    # manual and auto modes.
    stt_enabled = True  # session boots with STT wired; torn down in the boot below
    audio_output_enabled = True  # session default; muted at boot
    stt_timer: asyncio.TimerHandle | None = None
    publish_tasks: set[asyncio.Task[None]] = set()

    def _publish_state() -> None:
        payload = json.dumps(
            {"stt_enabled": stt_enabled, "audio_output": audio_output_enabled}
        )

        async def _send() -> None:
            try:
                await ctx.room.local_participant.publish_data(
                    payload, topic=SESSION_STATE_TOPIC, reliable=True
                )
            except Exception:
                logger.exception("failed to publish session state")

        task = asyncio.create_task(_send())
        publish_tasks.add(task)
        task.add_done_callback(publish_tasks.discard)

    def _cancel_stt_timer() -> None:
        nonlocal stt_timer
        if stt_timer is not None:
            stt_timer.cancel()
            stt_timer = None

    def _enable_stt() -> None:
        nonlocal stt_enabled
        _cancel_stt_timer()
        if stt_enabled:
            return
        agent.update_options(stt=stt)
        stt_enabled = True
        logger.info("STT enabled")
        _publish_state()

    def _disable_stt() -> None:
        nonlocal stt_enabled
        _cancel_stt_timer()
        if not stt_enabled:
            return
        agent.update_options(stt=None)
        stt_enabled = False
        logger.info("STT disabled (mic idle) to save cost")
        _publish_state()

    def _set_audio_input(enabled: bool) -> None:
        """Gate the mic and tie STT to it: enabled while listening, and scheduled for
        teardown STT_IDLE_TIMEOUT after the mic goes quiet (a grace period so quick,
        successive turns don't re-initialise STT)."""
        nonlocal stt_timer
        if enabled:
            _enable_stt()  # before opening the mic so STT is ready for the first words
            session.input.set_audio_enabled(True)
        else:
            session.input.set_audio_enabled(False)
            _cancel_stt_timer()
            if stt_enabled:
                loop = asyncio.get_running_loop()
                stt_timer = loop.call_later(settings.stt_idle_timeout, _disable_stt)

    def _set_audio_output(enabled: bool) -> None:
        """Toggle spoken (TTS) replies. Text replies are unaffected."""
        nonlocal audio_output_enabled
        if enabled == audio_output_enabled:
            return
        session.output.set_audio_enabled(enabled)
        audio_output_enabled = enabled
        logger.info("audio output %s", "enabled" if enabled else "disabled")
        _publish_state()

    async def _set_can_subscribe(identity: str, allow: bool) -> None:
        """Grant/revoke a participant's track-subscribe permission at runtime.

        The card connects with can_subscribe=False so an idle/text connection has no
        receive-audio transceiver (which on iOS grabs the audio session and stops the
        user's music). We allow subscribing only while spoken replies are on.
        """
        if not identity:
            return
        try:
            await ctx.api.room.update_participant(
                api.UpdateParticipantRequest(
                    room=ctx.room.name,
                    identity=identity,
                    permission=api.ParticipantPermission(
                        can_publish=True,
                        can_publish_data=True,
                        can_subscribe=allow,
                    ),
                )
            )
        except Exception:
            logger.exception("failed to update subscribe permission for %s", identity)

    def apply_mode(manual: bool) -> None:
        """Switch turn detection and gate the mic to match. STT follows the mic."""
        session.update_options(turn_detection="manual" if manual else turn_detector)
        if manual:
            session.clear_user_turn()
            _set_audio_input(False)  # idle until start_turn opens a turn
        else:
            _set_audio_input(True)  # auto: mic stays live so the model can detect turns

    # Boot dormant + text-only: mic gated, STT torn down now (no idle wait), TTS muted.
    session.input.set_audio_enabled(False)
    _disable_stt()
    _set_audio_output(False)
    _publish_state()

    @ctx.room.on("participant_connected")
    def _on_participant_connected(_participant: rtc.RemoteParticipant) -> None:
        # Re-assert state so a frontend that joins/reconnects mid-session sees it.
        _publish_state()

    async def _on_shutdown() -> None:
        _cancel_stt_timer()

    ctx.add_shutdown_callback(_on_shutdown)

    @ctx.room.local_participant.register_rpc_method("set_turn_mode")
    async def set_turn_mode(data: rtc.RpcInvocationData) -> str:
        manual = data.payload == "manual"
        logger.info("set turn mode: %s", "manual" if manual else "auto")
        apply_mode(manual)
        return "ok"

    @ctx.room.local_participant.register_rpc_method("set_audio_output")
    async def set_audio_output(data: rtc.RpcInvocationData) -> str:
        # payload "on"/"off" toggles spoken (TTS) replies; text replies still work.
        # Grant subscribe before enabling TTS; revoke after disabling it, so the client
        # only holds the audio session while replies play. No reconnect either way.
        on = data.payload == "on"
        if on:
            await _set_can_subscribe(data.caller_identity, True)
        _set_audio_output(on)
        if not on:
            await _set_can_subscribe(data.caller_identity, False)
        return "ok"

    @ctx.room.local_participant.register_rpc_method("start_turn")
    async def start_turn(data: rtc.RpcInvocationData) -> str:
        session.interrupt()
        session.clear_user_turn()
        # listen only to the participant who started the turn (multi-user rooms)
        session.room_io.set_participant(data.caller_identity)
        _set_audio_input(True)  # opens the mic and (re)enables STT
        return "ok"

    @ctx.room.local_participant.register_rpc_method("end_turn")
    async def end_turn(data: rtc.RpcInvocationData) -> str:
        _set_audio_input(False)  # gate the mic; STT tears down after the idle timeout
        session.commit_user_turn()
        return "ok"

    @ctx.room.local_participant.register_rpc_method("cancel_turn")
    async def cancel_turn(data: rtc.RpcInvocationData) -> str:
        _set_audio_input(False)
        session.clear_user_turn()
        logger.info("cancel turn")
        return "ok"


if __name__ == "__main__":
    cli.run_app(server)
