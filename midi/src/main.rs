use anyhow::Result;
use midir::{Ignore, MidiInput};
use std::io::stdin;

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
            let name_cloned = name.clone();
            
            let conn = midi_in.connect(
                &p,
                "midi_monitor_in",
                move |stamp, message, _| {
                    println!("[{}] Time: {}us, Data: {:02x?}", name_cloned, stamp, message);
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