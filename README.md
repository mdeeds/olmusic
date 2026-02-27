# OLMusic - OLMo Music Companion

## Prerequisites

This project is built with Rust. Before you begin, you'll need to install the Rust toolchain.

We assume you are developing on a Windows platform. Some modifications to these
instructions may be required for other development environments.

1.  **Install C++ Build Tools:** Rust's MSVC toolchain requires the [Microsoft 
C++ Build Tools](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools).
    -   Go to the Visual Studio Downloads page.
    -   Under "All Downloads" -> "Tools for Visual Studio", find "Build Tools for Visual Studio" and click Download.
    -   Run the installer. In the "Workloads" tab, select **"Desktop development with C++"** and then click "Install".

2.  **Install Rust:** If you don't have Rust installed, visit https://rustup.rs/ and follow the on-screen instructions. `rustup` should automatically detect your C++ Build Tools installation. The default options are sufficient. You may need to restart your terminal after installation.

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/mdeeds/olmusic.git
cd olmusic
```

## Demos

### English Language Inference

This demo runs a basic inference task using the OLMo model to generate text.

1.  Navigate to the demo directory:
    ```cmd
    cd inference\english
    ```

2.  Run the demo using Cargo:
    ```bash
    cargo run --release
    ```

If you get an error about `kernel32.dll` being unavailable, you may need
to install the developer tools as described above.



**Note:** The first time you run this command, Cargo will download and compile all the necessary dependencies, and the application will download the model weights from the Hugging Face Hub. This may take a significant amount of time and requires an internet connection. Subsequent runs will be much faster.