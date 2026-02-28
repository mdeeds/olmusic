use anyhow::Result;
use midir::{Ignore, MidiInput};
use std::io::{stdin, stdout, Write};
use std::sync::{Arc, Mutex};

struct MidiHandler {
    clock_counter: u32,
}

impl MidiHandler {
    fn new() -> Self {
        Self { clock_counter: 0 }
    }

    fn handle_message(&mut self, message: &[u8]) {
        // http://www.opensound.com/pguide/midi/midi5.html
        if message.len() == 0 {
            return; // Ignore empty messages.
        } else if message[0] == 0xf0 {
            return; // Ignore SysEx messages.
        } else if message[0] & 0xf0 == 0xb0 {
            // Juno 106 sends these when all keys are lifted.
            return; // Ignore Control Change messages.
        } else if message.len() == 1 && message[0] == 0xf8 {
            self.clock_counter += 1;
            if self.clock_counter >= 24 {
                print!("Q ");
                let _ = stdout().flush();
                self.clock_counter = 0;
            }
        } else if message[0] & 0xf0 == 0xf0 {
            return; // Ignore other system real-time messages (Active Sensing, etc).
        }
        // Interesting things start here        
        else if message.len() == 1 && message[0] == 0xfa {
            // Start of a sequence - reset clock counter.
            self.clock_counter = 0;
            print!("S ");
            let _ = stdout().flush();
        } else if message.len() == 3 && 
            (message[0] & 0xf0 == 0x90 && message[2] == 0) || 
            (message.len() == 3 && message[0] & 0xf0 == 0x80) {
          // Note off message (Note on with velocity 0) - print the note number.
          print!("O{:02x} ", message[1]);
          let _ = stdout().flush();
        } else if message.len() == 3 && message[0] & 0xf0 == 0x90 {
            // Note on message - print the note number and velocity.
            print!("N{:02x} v{:02x} ", message[1], message[2]);
            let _ = stdout().flush();
        } else {

            for byte in message {
                print!("{:02x}", byte);
            }
            print!(" ");
            let _ = stdout().flush();
        }
    }
}

fn main() -> Result<()> {
    // 1. Probe for available ports using a temporary client.
    // We ignore nothing (Sysex, Timing, Active Sensing) to ensure we see all traffic.
    let mut midi_in_probe = MidiInput::new("midi_monitor_probe")?;
    midi_in_probe.ignore(Ignore::None);
    
    let ports = midi_in_probe.ports();
    let port_names: Vec<String> = ports
        .iter()
        .map(|p| midi_in_probe.port_name(p).unwrap_or_else(|_| "Unknown".to_string()))
        .collect();

    if port_names.is_empty() {
        println!("No MIDI input ports found.");
        return Ok(());
    }

    println!("Found {} MIDI ports:", port_names.len());
    for name in &port_names {
        println!(" - {}", name);
    }

    // 2. Connect to each port.
    // We need to keep the input connections alive in a vector, otherwise they will be dropped and closed.
    let mut connections = Vec::new();
    let handler = Arc::new(Mutex::new(MidiHandler::new()));

    for name in port_names {
        // Create a new client for each port to allow simultaneous connections.
        // (midir consumes the client instance upon connection).
        let mut midi_in = MidiInput::new(&format!("monitor_{}", name))?;
        midi_in.ignore(Ignore::None);

        // We have to re-find the port by name because MidiInputPort is not safely transferable 
        // across MidiInput instances on all platforms/backends.
        let ports = midi_in.ports();
        let port = ports.into_iter()
            .find(|p| midi_in.port_name(p).ok().as_deref() == Some(&name));

        if let Some(p) = port {
            println!("Connecting to '{}'...", name);
            let handler = handler.clone();
            
            let conn = midi_in.connect(
                &p,
                "midi_monitor_in",
                move |_stamp, message, _| {
                    handler.lock().unwrap().handle_message(message);
                },
                (),
            ).map_err(|e| anyhow::anyhow!("Failed to connect to {}: {}", name, e))?;
            
            connections.push(conn);
        } else {
            eprintln!("Warning: Could not re-locate port '{}'", name);
        }
    }

    println!("\nListening for MIDI events... Press Enter to exit.");
    let mut input = String::new();
    stdin().read_line(&mut input)?;

    Ok(())
}