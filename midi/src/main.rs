use anyhow::Result;
use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use std::io::{stdin, stdout, Write};
use std::sync::{Arc, Mutex};

struct MidiSender {
    conn: Option<MidiOutputConnection>,
    clock_conn: Option<MidiInputConnection<()>>,
    channel: u8,
}

impl MidiSender {
    fn new(conn: Option<MidiOutputConnection>, channel: u8) -> Self {
        Self { conn, clock_conn: None, channel }
    }

    fn set_clock_input(&mut self, conn: MidiInputConnection<()>) {
        self.clock_conn = Some(conn);
    }

    fn handle_clock(&mut self) {
        // Listen for clock
    }
}

struct MidiHandler {
    clock_counter: u32,
    delta_clock_counter: u32,
    history: Vec<String>,
}

impl MidiHandler {
    fn new() -> Self {
        Self { 
            clock_counter: 0,
            delta_clock_counter: 0,
            history: Vec::new(),
        }
    }

    fn increment_clock(&mut self) -> Option<String> {
        self.clock_counter += 1;
        self.delta_clock_counter += 1;
        if self.clock_counter >= 24 {
            self.clock_counter = 0;
            self.delta_clock_counter = 0;
            Some("Q ".to_string())
        } else {
            None
        }
    }

    fn reset_clock(&mut self) {
        self.clock_counter = 0;
        self.delta_clock_counter = 0;
    }

    fn get_delta_time(&mut self) -> String {
        if self.delta_clock_counter > 0 {
            let s = format!("P{:02x} ", self.delta_clock_counter);
            self.delta_clock_counter = 0;
            s
        } else {
            String::new()
        }
    }

    fn handle_message(&mut self, message: &[u8]) -> Option<String> {
        let mut output = String::new();
        // http://www.opensound.com/pguide/midi/midi5.html
        if message.len() == 0 {
            return None; // Ignore empty messages.
        } else if message[0] == 0xf0 {
            return None; // Ignore SysEx messages.
        } else if message[0] & 0xf0 == 0xb0 {
            // Juno 106 sends these when all keys are lifted.
            return None; // Ignore Control Change messages.
        } else if message.len() == 1 && message[0] == 0xf8 {
            if let Some(s) = self.increment_clock() {
                output.push_str(&s);
            }
        }
        else if message.len() == 1 && message[0] == 0xfa {
            // Start of a sequence - reset clock counter.
            self.reset_clock();
            output.push_str("Start ");
        } else if message.len() == 1 && message[0] == 0xfc {
            // End of sequence.
            output.push_str("End ");
        } else if message[0] & 0xf0 == 0xf0 {
            return None; // Ignore other system real-time messages (Active Sensing, etc).
        } else if message.len() == 3 && 
            (message[0] & 0xf0 == 0x90 && message[2] == 0) || 
            (message.len() == 3 && message[0] & 0xf0 == 0x80) {
          // Note off message (Note on with velocity 0) - print the note number.
          output.push_str(&self.get_delta_time());
          output.push_str(&format!("O{:02x} ", message[1]));
        } else if message.len() == 3 && message[0] & 0xf0 == 0x90 {
            // Note on message - print the note number and velocity.
            output.push_str(&self.get_delta_time());
            output.push_str(&format!("N{:02x} v{:02x} ", message[1], message[2]));
        } else {
            output.push_str("M");
            for byte in message {
                output.push_str(&format!("{:02x}", byte));
            }
            output.push_str(" ");
        }

        if output.is_empty() {
            None
        } else {
            self.history.push(output.clone());
            Some(output)
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

    // Setup output
    let args: Vec<String> = std::env::args().collect();
    let output_port_name = args.iter()
        .find(|a| a.starts_with("--output="))
        .map(|a| a.trim_start_matches("--output="));
    let clock_port_name = args.iter()
        .find(|a| a.starts_with("--clock="))
        .map(|a| a.trim_start_matches("--clock="));

    let midi_out = MidiOutput::new("midi_out_client")?;
    let out_ports = midi_out.ports();
    let out_port = if let Some(name) = output_port_name {
        let p = out_ports.iter().find(|p| midi_out.port_name(p).ok().as_deref() == Some(name));
        if p.is_none() {
            println!("Output port '{}' not found.", name);
        }
        p
    } else {
        out_ports.get(0)
    };

    let out_conn = if let Some(p) = out_port {
        Some(midi_out.connect(p, "midi_out_conn").map_err(|e| anyhow::anyhow!("{}", e))?)
    } else {
        None
    };

    let sender = Arc::new(Mutex::new(MidiSender::new(out_conn, 1)));

    // Setup clock input
    if let Some(clock_name) = clock_port_name {
        let mut midi_in_clock = MidiInput::new("midi_clock_in")?;
        midi_in_clock.ignore(Ignore::None);
        let ports = midi_in_clock.ports();
        let clock_port = ports.iter().find(|p| midi_in_clock.port_name(p).ok().as_deref() == Some(clock_name));
        
        if let Some(p) = clock_port {
             let sender_weak = Arc::downgrade(&sender);
             let conn = midi_in_clock.connect(
                 p,
                 "midi_clock_conn",
                 move |_, message, _| {
                     if message.len() == 1 && message[0] == 0xf8 {
                         if let Some(sender_arc) = sender_weak.upgrade() {
                             sender_arc.lock().unwrap().handle_clock();
                         }
                     }
                 },
                 ()
             ).map_err(|e| anyhow::anyhow!("Failed to connect to clock {}: {}", clock_name, e))?;
             
             sender.lock().unwrap().set_clock_input(conn);
             println!("Connected to clock source '{}'", clock_name);
        } else {
             println!("Clock port '{}' not found.", clock_name);
        }
    }

    // 2. Connect to each port.
    // We need to keep the input connections alive in a vector, otherwise they will be dropped and closed.
    let mut connections = Vec::new();
    let handler = Arc::new(Mutex::new(MidiHandler::new()));

    for name in port_names {
        if Some(name.as_str()) == clock_port_name { continue; }

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
                    if let Some(s) = handler.lock().unwrap().handle_message(message) {
                        print!("{}", s);
                        let _ = stdout().flush();
                    }
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