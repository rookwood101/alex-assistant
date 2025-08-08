import asyncio
from types import SimpleNamespace

import pytest


class FakeInlineData:
    def __init__(self, data: bytes):
        self.data = data


class FakePart:
    def __init__(self, data: bytes):
        self.inline_data = FakeInlineData(data)


class FakeServerContent:
    def __init__(self, input_text: str = "", output_text: str = "", audio_bytes: bytes = b"", turn_complete: bool = False):
        self.input_transcription = SimpleNamespace(text=input_text) if input_text else None
        self.output_transcription = SimpleNamespace(text=output_text) if output_text else None
        self.model_turn = SimpleNamespace(parts=[FakePart(audio_bytes)]) if audio_bytes else None
        self.turn_complete = turn_complete


class FakeChunk:
    def __init__(self, server_content: FakeServerContent | None = None, tool_call=None):
        self.server_content = server_content
        self.tool_call = tool_call


class FakeSession:
    def __init__(self, script: list[FakeChunk]):
        self._script = script
        self._send_audio_count = 0
        self._send_text = []
        self._tool_responses = []

    async def send_realtime_input(self, *, audio=None, text: str | None = None):
        if audio is not None:
            self._send_audio_count += 1
        if text is not None:
            self._send_text.append(text)

    async def send_tool_response(self, *, function_responses):
        self._tool_responses.extend(function_responses)

    async def receive(self):
        for item in self._script:
            yield item
        # End of stream
        return


def make_two_turn_script():
    # One input transcription, one output with audio, then mark turn complete
    chunks = [
        FakeChunk(FakeServerContent(input_text="hello")),
        FakeChunk(FakeServerContent(output_text="hi.", audio_bytes=b"\x00\x00" * 24000)),
        FakeChunk(FakeServerContent(turn_complete=True)),
    ]
    return chunks


@pytest.mark.asyncio
async def test_single_conversation_flow(app_module: object):
    audio_input_queue = asyncio.Queue()
    audio_output_queue = asyncio.Queue()
    tasks: list[asyncio.Task] = []

    # Fake tools dict is empty for simplicity
    tools: dict[str, callable] = {}

    # Build a fake session with a simple script
    session = FakeSession(make_two_turn_script())

    # Seed input queue with a few frames so send_audio_to_gemini has something to forward
    for _ in range(3):
        audio_input_queue.put_nowait(b"\x00\x00" * app_module.porcupine.frame_length)

    await app_module.run_conversation(
        session=session,
        audio_input_queue=audio_input_queue,
        audio_output_queue=audio_output_queue,
        tasks=tasks,
        tools=tools,
        initial_text=None,
    )

    # After conversation, background gemini task should be cancelled and awaited
    # tasks list may still contain the completed task, but it should be done
    assert all(t.cancelled() or t.done() for t in tasks)

    # Output queue should be drained after turn_complete
    assert audio_output_queue.empty()


@pytest.mark.asyncio
async def test_two_consecutive_conversations_reset_and_cleanup(app_module: object, monkeypatch: pytest.MonkeyPatch):
    # Prepare queues and tasks
    audio_input_queue = asyncio.Queue()
    audio_output_queue = asyncio.Queue()
    tasks: list[asyncio.Task] = []

    # Track calls to reset_vad_iterator
    reset_calls = {"count": 0}

    def fake_reset():
        reset_calls["count"] += 1

    monkeypatch.setattr(app_module, "reset_vad_iterator", fake_reset)

    tools: dict[str, callable] = {}

    # Conversation 1
    session1 = FakeSession(make_two_turn_script())
    # Seed audio input for first session
    for _ in range(2):
        audio_input_queue.put_nowait(b"\x00\x00" * app_module.porcupine.frame_length)

    await app_module.run_conversation(
        session=session1,
        audio_input_queue=audio_input_queue,
        audio_output_queue=audio_output_queue,
        tasks=tasks,
        tools=tools,
        initial_text=None,
    )

    # Conversation 2
    session2 = FakeSession(make_two_turn_script())
    for _ in range(2):
        audio_input_queue.put_nowait(b"\x00\x00" * app_module.porcupine.frame_length)

    await app_module.run_conversation(
        session=session2,
        audio_input_queue=audio_input_queue,
        audio_output_queue=audio_output_queue,
        tasks=tasks,
        tools=tools,
        initial_text=None,
    )

    # reset_vad_iterator should be called once per conversation
    assert reset_calls["count"] == 2

    # Background tasks should be cancelled/done after each conversation
    assert all(t.cancelled() or t.done() for t in tasks)

    # Output queue should end empty
    assert audio_output_queue.empty()

    # echo_reference_buffer is maintained in output task in app; for Layer 2
    # we aren't running output task. Ensure the input queue didn't grow unbounded.
    # Allow a small number of leftover frames that the cancelled task may not consume.
    assert audio_input_queue.qsize() <= 4


@pytest.mark.asyncio
async def test_detect_wakeword_with_stubbed_porcupine(app_module: object, monkeypatch: pytest.MonkeyPatch):
    # Prepare queue and event
    audio_input_queue = asyncio.Queue()
    conversation_inactive = asyncio.Event()
    conversation_inactive.set()

    # Provide frames and stub porcupine.process to yield -1 then 0
    sequence = [-1, -1, -1, 0]

    def fake_process(_pcm):
        return sequence.pop(0) if sequence else -1

    monkeypatch.setattr(app_module.porcupine, "process", fake_process)

    # enqueue enough frames for the sequence
    for _ in range(4):
        audio_input_queue.put_nowait(b"\x00\x00" * app_module.porcupine.frame_length)

    result = await asyncio.wait_for(
        app_module.detect_wakeword(audio_input_queue, conversation_inactive), timeout=1.0
    )
    assert result is True


@pytest.mark.asyncio
async def test_output_echo_reference_buffer_bounded(app_module: object, monkeypatch: pytest.MonkeyPatch):
    # Force AEC enabled and provide fake audio device
    monkeypatch.setattr(app_module, "ENABLE_AEC", True)

    class _FakeStream:
        def stop_stream(self):
            return

        def close(self):
            return

        def write(self, data: bytes, exception_on_underflow: bool = False):
            return

    class _FakeAudio:
        def open(self, **kwargs):
            return _FakeStream()

    monkeypatch.setattr(app_module, "audio", _FakeAudio())

    audio_output_queue = asyncio.Queue()
    echo_reference_buffer: list[bytes] = []

    # Start the output task
    task = asyncio.create_task(app_module.output_audio(audio_output_queue, echo_reference_buffer))

    # Push more audio chunks than the buffer size
    for _ in range(app_module.echo_buffer_size + 200):
        audio_output_queue.put_nowait(b"\x00\x00" * 24000)

    # Allow the task to process
    await asyncio.sleep(0.05)

    # Cancel the task
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Buffer should be bounded
    assert len(echo_reference_buffer) <= app_module.echo_buffer_size


@pytest.mark.asyncio
async def test_repeat_two_conversations_ten_times(app_module: object, monkeypatch: pytest.MonkeyPatch):
    # Track reset calls across many iterations
    reset_calls = {"count": 0}

    def fake_reset():
        reset_calls["count"] += 1

    monkeypatch.setattr(app_module, "reset_vad_iterator", fake_reset)

    tools: dict[str, callable] = {}
    audio_output_queue = asyncio.Queue()

    for _ in range(10):
        audio_input_queue = asyncio.Queue()
        session = FakeSession(make_two_turn_script())
        # feed a couple of frames
        for _i in range(2):
            audio_input_queue.put_nowait(b"\x00\x00" * app_module.porcupine.frame_length)
        tasks: list[asyncio.Task] = []
        await app_module.run_conversation(
            session=session,
            audio_input_queue=audio_input_queue,
            audio_output_queue=audio_output_queue,
            tasks=tasks,
            tools=tools,
            initial_text=None,
        )
        assert all(t.cancelled() or t.done() for t in tasks)
        assert audio_output_queue.empty()

    # Expect 10 resets
    assert reset_calls["count"] == 10


