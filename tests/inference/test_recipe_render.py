"""Execute unchanged text-rendering Rust functions extracted from the pinned recipe.

This narrow differential does not claim full recipe/tool/image qualification.
The full crate needs uncached Cargo dependencies; this harness needs rustc only.
"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from .test_text import FIXTURES, official, tokenizer


def test_original_rust_text_rendering(tokenizer, tmp_path):
    root = os.environ.get("ELPIS_DEEPSEEK_RECIPE")
    if not root:
        pytest.skip("ELPIS_DEEPSEEK_RECIPE required for original Rust template differential")
    rustc = shutil.which("rustc")
    if rustc is None:
        pytest.skip("rustc required for original template differential")
    base = Path(root) / "deepseek-recipe-encoding/src/v4"
    source = (base / "mod.rs").read_text()
    v41 = (base / "dsv41.rs").read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == "0a4577a216bcbddabd5bcf969cabd75060deb79e643548f8a04bda1d76e4a982"
    assert hashlib.sha256(v41.encode()).hexdigest() == "3408554e8a4ade05e034231cab8f87efcee7459d7fdd28fbb926f652e3b56786"
    # Text function bodies are copied byte-for-byte at test time, not ported to
    # Python. Only their data-type shell and unavailable tool path are stubbed.
    render = source[source.index("fn render_message("):source.index("impl<T: EncodingV4> PromptEncoding")]
    effort = v41[v41.index("fn reasoning_effort_template("):v41.index("/// Prompt rendering")]
    shell = r'''
#![allow(dead_code)]
#[derive(Clone, Copy)] enum ReasoningEffort { Low, High, Xhigh, Max }
struct ToolCall;
enum InputMessage {
 System { content: String }, User { content: String },
 Assistant { content: String, reasoning_content: Option<String>, tool_calls: Option<Vec<ToolCall>> },
 LatestReminder { content: String }, Tool { content: String },
}
trait EncodingV4 {
 fn render_reasoning_effort(&self, index: usize, thinking: bool, effort: Option<ReasoningEffort>) -> String;
 fn system_token(&self) -> &'static str;
 fn supports_mid_conversation_system(&self) -> bool;
}
struct Encoder;
impl EncodingV4 for Encoder {
 fn render_reasoning_effort(&self, i: usize, t: bool, e: Option<ReasoningEffort>) -> String {
   if i == 0 && t { reasoning_effort_template(e) } else { String::new() }
 }
 fn system_token(&self) -> &'static str { "<｜System｜>" }
 fn supports_mid_conversation_system(&self) -> bool { true }
}
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
const THINKING_START_TOKEN: &str = "<think>";
const THINKING_END_TOKEN: &str = "</think>";
const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
fn render_tool_calls(_: &impl EncodingV4, _: &[ToolCall]) -> String { panic!("tools outside text differential") }
fn tool_calls_template(_: &impl EncodingV4, _: &str) -> String { panic!("tools outside text differential") }
'''
    def string(s):
        assert '"' not in s and "\\" not in s
        return '"' + s + '".to_string()'
    cases = []
    for thinking, level, messages, expected in FIXTURES:
        entries = []
        for message in messages:
            role = message.role.title()
            fields = "content: " + string(message.content)
            if message.role == "assistant":
                fields += ", reasoning_content: " + ("None" if message.reasoning is None
                                                     else "Some(" + string(message.reasoning) + ")")
                fields += ", tool_calls: None"
            entries.append(f"InputMessage::{role} {{ {fields} }}")
        rs_level = "None" if level is None else "Some(ReasoningEffort::" + level.title() + ")"
        cases.append("{ let messages = vec![" + ",".join(entries) + "];"
                     'let mut prompt = "<｜begin▁of▁sentence｜>".to_string();'
                     f"for i in 0..messages.len() {{ prompt += &render_message(&Encoder, &messages, i, {str(thinking).lower()}, {rs_level}); }}"
                     'prompt += "<｜Assistant｜>"; prompt += ' + ('"<think>";' if thinking else '"</think>";')
                     + 'for b in prompt.as_bytes() { print!("{:02x}", b); } println!(); }')
    program = tmp_path / "oracle.rs"
    program.write_text(shell + effort + render + "fn main() {" + "\n".join(cases) + "}")
    executable = tmp_path / "oracle"
    compiled = subprocess.run([rustc, str(program), "-o", str(executable)], capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    prompts = subprocess.check_output([str(executable)], text=True).splitlines()
    assert len(prompts) == len(FIXTURES)
    for encoded, (thinking, level, messages, expected) in zip(prompts, FIXTURES):
        rendered = bytes.fromhex(encoded).decode()
        assert rendered == expected == tokenizer.render_chat(messages, thinking=thinking, effort=level)
        assert tokenizer.encode(rendered) == tokenizer.encode_chat(messages, thinking=thinking, effort=level)
