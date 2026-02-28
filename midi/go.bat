cls
mkdir %CD%\tmp_build 2>nul
set TEMP=%CD%\tmp_build
set TMP=%CD%\tmp_build
REM cargo clean
cargo run --release -- --output=H4MIDI-WC --clock=RC-505
cargo run --release -- --output=H4MIDI-WC
