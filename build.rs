//! Embeds the lmbrrr git revision and the candle fork pin into the binary so
//! every benchmark/report row records exactly which code produced it.

use std::process::Command;

fn main() {
    let git_rev = Command::new("git")
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    println!(
        "cargo:rustc-env=LMBRRR_GIT_REV={}{}",
        git_rev,
        if dirty { "-dirty" } else { "" }
    );

    let candle_pin = std::fs::read_to_string("Cargo.lock")
        .ok()
        .and_then(|lock| {
            lock.split("[[package]]")
                .find(|pkg| pkg.contains("name = \"candle-core\""))
                .and_then(|pkg| {
                    pkg.lines()
                        .find(|l| l.starts_with("source = ") && l.contains("rev="))
                        .and_then(|l| l.split("rev=").nth(1))
                        .map(|tail| {
                            tail.trim_end_matches('"')
                                .split('#')
                                .next()
                                .unwrap_or("unknown")
                                .to_string()
                        })
                })
        })
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=LMBRRR_CANDLE_PIN={candle_pin}");

    println!("cargo:rerun-if-changed=Cargo.lock");
    println!("cargo:rerun-if-changed=.git/HEAD");
}
