"""Normalizers and the thinking format for the SFT mixture.

These cover the tasks added on top of the original seven -- shell, rewrite and
writing -- plus the plain-text scratchpad, which has no special token to anchor
it and so is only as reliable as the string contract in sft.py.
"""

import json
import random
from collections import Counter

import pytest

from nanollm.data.sft import (ANSWER_PREFIX, DOLLY_TASKS, NO_ROBOTS_TASKS,
                              TASK_HEAD_COMMAND_CAP, TASK_REPEAT_CAPS,
                              TASK_WEIGHTS, THINK_PREFIX,
                              MAX_TASK_REPEATS, SFTDataset, _cap_head_command,
                              _from_coedit, _from_dolly, _from_ecqa,
                              _from_gsm8k, _head_command,
                              _from_metamath, _from_nl2bash, _from_nl2sh,
                              _from_no_robots, _split_worked_solution,
                              _think_reply, split_thinking)


@pytest.fixture
def rng():
    return random.Random(0)


def unpack(result):
    """Normalizers return either messages or (messages, task)."""
    if isinstance(result, tuple):
        return result
    return result, None


# ---------------------------------------------------------------------------
# Mixture bookkeeping
# ---------------------------------------------------------------------------

def test_task_weights_sum_to_one():
    assert sum(TASK_WEIGHTS.values()) == pytest.approx(1.0)


def test_every_routed_task_has_a_weight():
    """A category routed to a task with no weight is data thrown away: the
    rows land in the cache and _build_mixture never draws them."""
    for task in set(NO_ROBOTS_TASKS.values()) | set(DOLLY_TASKS.values()):
        assert task in TASK_WEIGHTS, f"{task} is routed to but never weighted"


def test_mixture_hits_every_target_share_at_the_default_epoch_size():
    """Pool sizes are roughly what the configured sources deliver. If a task
    caps at 200k examples its share silently drops below target, which is the
    failure the epoch-size comment in sft.py is about."""
    pools = {"chat": 400_000, "extractive_qa": 70_000, "writing": 13_300,
             "rewrite": 70_000, "math": 122_000, "summarization": 54_000,
             # shell is post-cap: TASK_HEAD_COMMAND_CAP thins the `find`
             # monoculture, which is what the task actually ships now.
             "shell": 32_000, "reasoning": 20_600, "instruct": 5_000,
             "sql": 180_000, "smalltalk": 2_200}
    caps = {t: pools[t] * TASK_REPEAT_CAPS.get(t, MAX_TASK_REPEATS)
            for t in pools}
    alloc = SFTDataset._allocate(dict(TASK_WEIGHTS), caps, 200_000,
                                 sum(TASK_WEIGHTS.values()))
    total = sum(alloc.values())
    for task, weight in TASK_WEIGHTS.items():
        assert alloc[task] / total == pytest.approx(weight, abs=0.005)
        assert alloc[task] < caps[task], f"{task} is capped at 200k examples"


# ---------------------------------------------------------------------------
# Thinking
# ---------------------------------------------------------------------------

def test_think_reply_round_trips():
    reply = _think_reply("Two plus two is four.", "4")
    thinking, answer = split_thinking(reply)
    assert thinking == "Two plus two is four."
    assert answer == "4"


def test_think_reply_falls_back_when_the_rationale_is_missing():
    """Rule 1: never invent working. No rationale means a bare answer, not a
    Thinking: block wrapped around the answer restated."""
    assert _think_reply("", "4") == "4"
    assert _think_reply("   ", "4") == "4"
    assert _think_reply("4", "4") == "4"          # rationale is just the answer
    assert THINK_PREFIX not in _think_reply("tiny", "4")


def test_think_reply_needs_an_answer():
    assert _think_reply("Some working here that is long enough.", "") is None


def test_split_thinking_passes_plain_replies_through():
    assert split_thinking("just an answer") == ("", "just an answer")
    assert split_thinking(None) == ("", "")


def test_split_thinking_tolerates_a_truncated_scratchpad():
    """A reply cut off by max_new_tokens mid-thought has no Answer: line."""
    thinking, answer = split_thinking(f"{THINK_PREFIX} halfway through the")
    assert thinking == "halfway through the"
    assert answer == ""


def test_split_worked_solution_rejects_a_conclusion_with_no_working():
    assert _split_worked_solution("The answer is 4.") == (None, None)
    assert _split_worked_solution("") == (None, None)
    assert _split_worked_solution("no closing statement here") == (None, None)


def test_split_worked_solution_finds_the_trailing_answer():
    steps, final = _split_worked_solution(
        "She has 5 apples and buys 3 more, so 5 + 3 = 8. The answer is: 8")
    assert final == "8"
    assert "5 + 3 = 8" in steps
    assert "answer is" not in steps.lower()


# ---------------------------------------------------------------------------
# Maths sources
# ---------------------------------------------------------------------------

def test_gsm8k_produces_both_modes(rng):
    row = {"question": "Weng earns $12 an hour. How much for 50 minutes?",
           "answer": "Weng earns 12/60 = <<12/60=0.2>>$0.2 per minute.\n"
                     "For 50 minutes she earned 0.2 x 50 = <<0.2*50=10>>$10.\n"
                     "#### 10"}
    replies = {unpack(_from_gsm8k(row, rng))[0][-1]["content"]
               for _ in range(60)}
    assert any(r.startswith(THINK_PREFIX) for r in replies)
    assert "10" in replies, "the direct-answer contrast never appears"
    for reply in replies:
        assert "<<" not in reply, "calculator annotation leaked through"
        assert "####" not in reply


def test_metamath_strips_gsm8k_markup_without_eating_the_answer(rng):
    """The #### strip must not be greedy to end-of-string: metamath writes
    '... #### 8 The answer is: 8', and a DOTALL strip removes the very line
    _split_worked_solution keys on."""
    row = {"type": "GSM_AnsAug", "query": "How many apples?",
           "response": "She has 5 and buys 3, so 5 + 3 = <<5+3=8>>8.\n"
                       "#### 8\nThe answer is: 8"}
    replies = {unpack(_from_metamath(row, rng))[0][-1]["content"]
               for _ in range(60)}
    assert any(r.startswith(THINK_PREFIX) for r in replies), \
        "the answer marker was stripped along with ####"
    for reply in replies:
        assert "####" not in reply and "<<" not in reply


def test_metamath_drops_the_competition_maths_subset(rng):
    """MATH_* rows are where metamath's LaTeX lives, and are problems a 169M
    model cannot do -- see METAMATH_KEEP_PREFIX."""
    latex = {"type": "MATH_AnsAug",
             "query": r"Evaluate $\sqrt{2 - \sqrt{2}}$.",
             "response": r"We have $\sqrt{2} \approx 1.41$. The answer is: 0.77"}
    assert _from_metamath(latex, rng) is None
    assert _from_metamath(dict(latex, type="MATH_FOBAR"), rng) is None
    assert _from_metamath(dict(latex, type="GSM_AnsAug"), rng) is not None


def test_metamath_keeps_unsplittable_solutions_whole(rng):
    """No closing statement means no scratchpad -- but the row is still
    worth training on, so it must not be dropped."""
    row = {"type": "GSM_Rephrased", "query": "Why is the sky blue?",
           "response": "Shorter wavelengths scatter more in the atmosphere, "
                       "which is why the sky looks blue to us."}
    messages, _ = unpack(_from_metamath(row, rng))
    assert THINK_PREFIX not in messages[-1]["content"]
    assert "scatter" in messages[-1]["content"]


# ---------------------------------------------------------------------------
# Reasoning with a real rationale
# ---------------------------------------------------------------------------

def _ecqa_row():
    return {"q_text": "What might a person see at a brutal killing?",
            "q_op1": "bloody mess", "q_op2": "pleasure", "q_op3": "cake",
            "q_op4": "guilt", "q_op5": "prison", "q_ans": "bloody mess",
            "taskA_pos": "A bloody mess is covered or stained with blood, "
                         "which is what such a scene would look like."}


def test_ecqa_labels_the_right_option(rng):
    for _ in range(20):
        messages, _ = unpack(_from_ecqa(_ecqa_row(), rng))
        prompt, reply = messages[0]["content"], messages[-1]["content"]
        assert "A) bloody mess" in prompt
        _, answer = split_thinking(reply)
        assert answer == "A) bloody mess"


def test_ecqa_produces_both_modes(rng):
    replies = [unpack(_from_ecqa(_ecqa_row(), rng))[0][-1]["content"]
               for _ in range(80)]
    assert any(r.startswith(THINK_PREFIX) for r in replies)
    assert any(not r.startswith(THINK_PREFIX) for r in replies)


def test_ecqa_drops_rows_whose_answer_is_not_an_option(rng):
    row = _ecqa_row()
    row["q_ans"] = "something else entirely"
    assert _from_ecqa(row, rng) is None


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def test_nl2bash_and_nl2sh_render_the_bare_command(rng):
    a, _ = unpack(_from_nl2bash(
        {"nl_command": "list files", "bash_code": "ls -la"}, rng))
    b, _ = unpack(_from_nl2sh({"nl": "free space", "bash": "df -h"}, rng))
    assert a[-1]["content"] == "ls -la"
    assert b[-1]["content"] == "df -h"
    assert "list files" in a[0]["content"]


@pytest.mark.parametrize("command", [
    "rm -rf /",
    "sudo rm -rf / --no-preserve-root",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    ":(){ :|:& };:",
    "shutdown -h now",
    "chmod -R 777 /",
])
def test_destructive_commands_are_dropped(command, rng):
    assert _from_nl2sh({"nl": "do a thing", "bash": command}, rng) is None


@pytest.mark.parametrize("command", [
    "rm -rf ./build",
    "rm -f notes.txt",
    "dd if=image.iso of=/dev/null",
    "find . -name '*.pyc' -delete",
])
def test_ordinary_commands_survive_the_filter(command, rng):
    assert _from_nl2sh({"nl": "do a thing", "bash": command}, rng) is not None


def test_multiline_scripts_are_dropped(rng):
    assert _from_nl2bash(
        {"nl_command": "x", "bash_code": "cd /tmp\nls"}, rng) is None


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

def test_coedit_replaces_the_glued_on_instruction(rng):
    row = {"task": "paraphrase", "src": "Rewrite this sentence: the cat sat",
           "tgt": "the cat was sitting"}
    messages, _ = unpack(_from_coedit(row, rng))
    prompt = messages[0]["content"]
    assert "the cat sat" in prompt
    assert "Rewrite this sentence:" not in prompt, "upstream prefix leaked"
    assert messages[-1]["content"] == "the cat was sitting"


def test_coedit_keeps_text_whose_colon_is_not_an_instruction(rng):
    """A long head before the first ': ' is part of the text, not a prefix."""
    body = ("In the study the authors noted a long and involved series of "
            "observations across many sites: the results were mixed")
    messages, _ = unpack(_from_coedit(
        {"task": "clarity", "src": body, "tgt": "The results were mixed."}, rng))
    assert body in messages[0]["content"]


def test_coedit_prompt_matches_the_edit_type(rng):
    prompts = {unpack(_from_coedit(
        {"task": "gec", "src": "x: he go", "tgt": "he goes"}, rng))[0][0]["content"]
        for _ in range(40)}
    assert all("grammar" in p.lower() or "grammatical" in p.lower()
               for p in prompts)


# ---------------------------------------------------------------------------
# Per-row task routing
# ---------------------------------------------------------------------------

def test_no_robots_routes_generation_to_writing(rng):
    row = {"category": "Generation",
           "messages": [{"role": "user", "content": "Write me a poem."},
                        {"role": "assistant", "content": "Roses are red."}]}
    _, task = unpack(_from_no_robots(row, rng))
    assert task == "writing"


def test_no_robots_chat_keeps_the_repo_task(rng):
    row = {"category": "Chat",
           "messages": [{"role": "user", "content": "Hi there."},
                        {"role": "assistant", "content": "Hello!"}]}
    _, task = unpack(_from_no_robots(row, rng))
    assert task is None, "an unrouted category must fall back to the manifest"


def test_dolly_routes_creative_writing(rng):
    row = {"category": "creative_writing", "instruction": "Write a limerick.",
           "context": "", "response": "There once was a model from Nantucket."}
    _, task = unpack(_from_dolly(row, rng))
    assert task == "writing"


def test_dolly_still_drops_contextless_closed_qa(rng):
    row = {"category": "closed_qa", "instruction": "Who won in 1994?",
           "context": "", "response": "Brazil."}
    assert _from_dolly(row, rng) is None


# -- command monoculture ----------------------------------------------------
# nl2bash/nl2sh are `find` corpora more than shell corpora; without a cap the
# shell weight buys find-flag trivia instead of the everyday commands.

def _shell_row(command):
    return {"task": "shell", "source": "nl2bash",
            "messages": json.dumps([{"role": "user", "content": "do a thing"},
                                    {"role": "assistant", "content": command}])}


def test_head_command_reads_the_command_not_the_prose():
    assert _head_command(_shell_row("find / -name '*.c'")["messages"]) == "find"
    assert _head_command(_shell_row("ls -la /tmp")["messages"]) == "ls"
    # A sudo prefix says nothing about what the row teaches.
    assert _head_command(_shell_row("sudo apt install vim")["messages"]) == "apt"
    assert _head_command("not json") == ""


def test_cap_head_command_thins_the_dominant_command():
    rows = [_shell_row(f"find / -name f{i}") for i in range(90)]
    rows += [_shell_row(f"ls -la dir{i}") for i in range(10)]
    kept = _cap_head_command(rows, random.Random(0), caps={"shell": 0.15})
    heads = Counter(_head_command(r["messages"]) for r in kept)
    assert heads["find"] == 15           # 15% of the 100 rows it started from
    assert heads["ls"] == 10             # under the cap, so untouched


def test_cap_head_command_leaves_unlisted_tasks_alone():
    rows = [{"task": "sql", "source": "x",
             "messages": json.dumps([{"role": "user", "content": "q"},
                                     {"role": "assistant", "content": "SELECT 1"}])}
            for _ in range(50)]
    assert _cap_head_command(rows, random.Random(0), caps={"shell": 0.15}) == rows


def test_shell_is_the_task_that_needs_the_cap():
    assert "shell" in TASK_HEAD_COMMAND_CAP
    assert 0 < TASK_HEAD_COMMAND_CAP["shell"] < 1
