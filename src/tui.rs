//! Interactive output for the `run` command: the <think>-tag stream parser,
//! the plain-stdout reasoning renderer, and the ratatui live view.

use std::io::{Stdout, Write};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::{
    cursor::{Hide, Show},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    text::Line,
    widgets::{Block, BorderType, Borders, Paragraph, Wrap},
    Terminal,
};

use crate::generate::{tokens_per_second, GenerationStats};

#[derive(Clone, Debug)]
pub struct ReasoningParts {
    pub raw_text: String,
    pub reasoning_text: String,
    pub answer_text: String,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum TextChannel {
    Reasoning,
    Answer,
}

impl TextChannel {
    pub fn label(self) -> &'static str {
        match self {
            Self::Reasoning => "Reasoning",
            Self::Answer => "Answer",
        }
    }
}

#[derive(Clone, Debug)]
pub enum TextEvent {
    Text(TextChannel, String),
}

#[derive(Clone, Debug)]
pub struct ReasoningTagParser {
    mode: TextChannel,
    pending: String,
}

impl ReasoningTagParser {
    pub fn new(mode: TextChannel) -> Self {
        Self {
            mode,
            pending: String::new(),
        }
    }

    pub fn feed(&mut self, text: &str) -> Vec<TextEvent> {
        self.pending.push_str(text);
        self.drain(false)
    }

    pub fn finish(&mut self) -> Vec<TextEvent> {
        self.drain(true)
    }

    fn drain(&mut self, final_chunk: bool) -> Vec<TextEvent> {
        const THINK_START: &str = "<think>";
        const THINK_END: &str = "</think>";

        let mut events = Vec::new();
        loop {
            let tag = match self.mode {
                TextChannel::Answer => THINK_START,
                TextChannel::Reasoning => THINK_END,
            };
            if let Some(idx) = self.pending.find(tag) {
                if idx > 0 {
                    events.push(TextEvent::Text(self.mode, self.pending[..idx].to_string()));
                }
                self.pending.drain(..idx + tag.len());
                self.mode = match self.mode {
                    TextChannel::Answer => TextChannel::Reasoning,
                    TextChannel::Reasoning => TextChannel::Answer,
                };
                continue;
            }

            if final_chunk {
                if !self.pending.is_empty() {
                    events.push(TextEvent::Text(
                        self.mode,
                        std::mem::take(&mut self.pending),
                    ));
                }
                break;
            }

            let keep = tag.len().saturating_sub(1);
            let emit_len = safe_prefix_len(&self.pending, keep);
            if emit_len == 0 {
                break;
            }
            let text = self.pending[..emit_len].to_string();
            self.pending.drain(..emit_len);
            events.push(TextEvent::Text(self.mode, text));
        }
        events
    }
}

#[derive(Debug)]
pub struct ReasoningRenderer {
    parser: ReasoningTagParser,
    active: Option<TextChannel>,
}

impl ReasoningRenderer {
    pub fn new(initial_channel: TextChannel) -> Self {
        Self {
            parser: ReasoningTagParser::new(initial_channel),
            active: None,
        }
    }

    pub fn write_chunk(&mut self, text: &str) -> Result<()> {
        for event in self.parser.feed(text) {
            self.render(event)?;
        }
        std::io::stdout().flush().ok();
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        for event in self.parser.finish() {
            self.render(event)?;
        }
        if self.active.is_some() {
            println!();
        }
        Ok(())
    }

    fn render(&mut self, event: TextEvent) -> Result<()> {
        match event {
            TextEvent::Text(_, text) if text.is_empty() => {}
            TextEvent::Text(channel, text) => {
                if self.active != Some(channel) {
                    if self.active.is_some() {
                        println!("\n");
                    }
                    println!("{}:", channel.label());
                    self.active = Some(channel);
                }
                print!("{text}");
            }
        }
        Ok(())
    }
}

pub struct TuiOutput {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    parser: ReasoningTagParser,
    reasoning_text: String,
    answer_text: String,
    prompt_tokens: usize,
    max_new_tokens: usize,
    last_draw: Instant,
}

impl TuiOutput {
    pub fn new(
        prompt_tokens: usize,
        max_new_tokens: usize,
        initial_channel: TextChannel,
    ) -> Result<Self> {
        enable_raw_mode().context("enable terminal raw mode")?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, Hide).context("enter terminal UI")?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend).context("create terminal UI")?;
        terminal.clear().context("clear terminal UI")?;

        let mut output = Self {
            terminal,
            parser: ReasoningTagParser::new(initial_channel),
            reasoning_text: String::new(),
            answer_text: String::new(),
            prompt_tokens,
            max_new_tokens,
            last_draw: Instant::now(),
        };
        output.draw(0, Duration::ZERO, None)?;
        Ok(output)
    }

    pub fn write_chunk(
        &mut self,
        text: &str,
        generated: usize,
        decode_elapsed: Duration,
        prefill_elapsed: Duration,
    ) -> Result<()> {
        let events = self.parser.feed(text);
        self.append_events(events);
        if self.last_draw.elapsed() >= Duration::from_millis(80) || generated == self.max_new_tokens
        {
            self.draw(generated, decode_elapsed, Some(prefill_elapsed))?;
        }
        Ok(())
    }

    pub fn finish(mut self, stats: &GenerationStats) -> Result<ReasoningParts> {
        let events = self.parser.finish();
        self.append_events(events);
        self.draw(
            stats.generated_tokens,
            stats.decode_elapsed,
            Some(stats.prefill_elapsed),
        )?;
        Ok(ReasoningParts {
            raw_text: format!("{}{}", self.reasoning_text, self.answer_text),
            reasoning_text: self.reasoning_text.clone(),
            answer_text: self.answer_text.clone(),
        })
    }

    fn append_events(&mut self, events: Vec<TextEvent>) {
        for event in events {
            match event {
                TextEvent::Text(TextChannel::Reasoning, text) => {
                    self.reasoning_text.push_str(&text)
                }
                TextEvent::Text(TextChannel::Answer, text) => self.answer_text.push_str(&text),
            }
        }
    }

    fn draw(
        &mut self,
        generated: usize,
        decode_elapsed: Duration,
        prefill_elapsed: Option<Duration>,
    ) -> Result<()> {
        let prompt_tokens = self.prompt_tokens;
        let max_output_tokens = self.max_new_tokens;
        let total_tokens = prompt_tokens + generated;
        let max_total_tokens = prompt_tokens + max_output_tokens;
        let prefill_rate = prefill_elapsed
            .map(|elapsed| format!("{:.2}", tokens_per_second(prompt_tokens, elapsed)))
            .unwrap_or_else(|| "prefilling".to_string());
        let output_rate = tokens_per_second(generated, decode_elapsed);
        let metrics = vec![
            Line::from(format!(
                "prefill: {prefill_rate} tok/s | output: {output_rate:.2} tok/s"
            )),
            Line::from(format!(
                "prompt tokens: {prompt_tokens} | output tokens: {generated} / {max_output_tokens} | total tokens: {total_tokens} / {max_total_tokens}"
            )),
        ];
        let reasoning = self.reasoning_text.clone();
        let answer = self.answer_text.clone();

        self.terminal
            .draw(|frame| {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(4),
                        Constraint::Percentage(45),
                        Constraint::Percentage(55),
                    ])
                    .split(frame.area());

                let metrics_widget = Paragraph::new(metrics).block(
                    Block::default()
                        .title("Metrics")
                        .borders(Borders::ALL)
                        .border_type(BorderType::Plain),
                );
                frame.render_widget(metrics_widget, chunks[0]);

                let reasoning_scroll = text_scroll(&reasoning, chunks[1].height);
                let reasoning_widget = Paragraph::new(reasoning)
                    .block(
                        Block::default()
                            .title("Reasoning")
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain),
                    )
                    .wrap(Wrap { trim: false })
                    .scroll((reasoning_scroll, 0));
                frame.render_widget(reasoning_widget, chunks[1]);

                let answer_scroll = text_scroll(&answer, chunks[2].height);
                let answer_widget = Paragraph::new(answer)
                    .block(
                        Block::default()
                            .title("Answer")
                            .borders(Borders::ALL)
                            .border_type(BorderType::Plain),
                    )
                    .wrap(Wrap { trim: false })
                    .scroll((answer_scroll, 0));
                frame.render_widget(answer_widget, chunks[2]);
            })
            .context("draw terminal UI")?;
        self.last_draw = Instant::now();
        Ok(())
    }
}

impl Drop for TuiOutput {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), Show, LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

pub fn split_reasoning_text(raw_text: &str, enable_thinking: bool) -> ReasoningParts {
    let initial_channel = if enable_thinking {
        TextChannel::Reasoning
    } else {
        TextChannel::Answer
    };
    let mut parser = ReasoningTagParser::new(initial_channel);
    let mut reasoning_text = String::new();
    let mut answer_text = String::new();
    for event in parser.feed(raw_text).into_iter().chain(parser.finish()) {
        match event {
            TextEvent::Text(TextChannel::Reasoning, text) => reasoning_text.push_str(&text),
            TextEvent::Text(TextChannel::Answer, text) => answer_text.push_str(&text),
        }
    }
    ReasoningParts {
        raw_text: raw_text.to_string(),
        reasoning_text,
        answer_text,
    }
}

pub fn print_reasoning_parts(parts: &ReasoningParts) -> Result<()> {
    if !parts.reasoning_text.trim().is_empty() {
        println!("Reasoning:");
        println!("{}", parts.reasoning_text.trim_end());
        println!();
    }
    if !parts.answer_text.trim().is_empty() {
        println!("Answer:");
        println!("{}", parts.answer_text.trim_end());
    }
    std::io::stdout().flush().ok();
    Ok(())
}

pub fn safe_prefix_len(text: &str, keep_bytes: usize) -> usize {
    let mut idx = text.len().saturating_sub(keep_bytes);
    while idx > 0 && !text.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

pub fn text_scroll(text: &str, height: u16) -> u16 {
    let visible_lines = height.saturating_sub(2).max(1) as usize;
    let line_count = text.lines().count().max(1);
    line_count.saturating_sub(visible_lines) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_reasoning_text_removes_think_tags() {
        let text = split_reasoning_text("<think>scratch</think>final", false);
        assert_eq!(text.raw_text, "<think>scratch</think>final");
        assert_eq!(text.reasoning_text, "scratch");
        assert_eq!(text.answer_text, "final");
    }

    #[test]
    fn split_reasoning_text_can_start_inside_think_block() {
        let text = split_reasoning_text("scratch</think>final", true);
        assert_eq!(text.reasoning_text, "scratch");
        assert_eq!(text.answer_text, "final");
    }

    #[test]
    fn parser_handles_split_reasoning_tag() {
        let mut parser = ReasoningTagParser::new(TextChannel::Answer);
        assert!(parser.feed("<thi").is_empty());
        let events = parser.feed("nk>scratch</think>final");
        let mut reasoning = String::new();
        let mut answer = String::new();
        for event in events.into_iter().chain(parser.finish()) {
            match event {
                TextEvent::Text(TextChannel::Reasoning, text) => reasoning.push_str(&text),
                TextEvent::Text(TextChannel::Answer, text) => answer.push_str(&text),
            }
        }
        assert_eq!(reasoning, "scratch");
        assert_eq!(answer, "final");
    }
}
