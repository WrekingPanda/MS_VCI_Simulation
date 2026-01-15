# MS_VCI_Simulation

A traffic simulation project using Eclipse SUMO for creating a digital twin of Porto's Via de Cintura Interna (VCI) with Vehicle-to-Infrastructure Communication capabilities.

## Project Overview

This repository contains a SUMO-based traffic simulation framework for the Via de Cintura Interna (VCI), Porto's inner ring road. The simulation leverages SUMO's TraCI (Traffic Control Interface) API to enable real-time interaction between the simulation and external Python scripts, allowing for dynamic traffic control, infrastructure communication, and data collection.

## Features

- Traffic flow simulation using Eclipse SUMO
- Python-based control scripts via TraCI interface
- Configurable simulation parameters
- Data output for traffic analysis

## Requirements

- Python 3.11.14
- pandas 2.3.3
- numpy 2.3.5
- Eclipse SUMO 1.24.0 (includes traci and sumolib)

## Installation

```bash
# Clone repository
git clone https://github.com/WrekingPanda/MS_VCI_Simulation.git
cd MS_VCI_Simulation

# Install Python dependencies
pip install pandas==2.3.3 numpy==2.3.5
```

### Install SUMO 1.24.0

**Linux:**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

**Windows:** Download from [eclipse.dev/sumo](https://eclipse.dev/sumo/)

**macOS:**
```bash
brew install sumo
```

## Setup

**Before running any script, verify the paths inside each `.py` file:**

```python
os.environ['SUMO_HOME'] = '/usr/share/sumo'  # Update to your path
sumo_binary = checkBinary('sumo-gui')        # or 'sumo'
```

**Common SUMO paths:**
- Linux: `/usr/share/sumo`
- Windows: `C:\Program Files (x86)\Eclipse\Sumo`
- macOS: `/opt/homebrew/share/sumo`

Find your path with:
```bash
which sumo        # Linux/macOS
where sumo        # Windows
```

## Usage

```bash
python <script_name>.py
```

Use `sumo-gui` in scripts for visualization, or `sumo` for headless mode.

## Project Structure

```
MS_VCI_Simulation/
├── *.py          # Python scripts
├── *.sumocfg     # SUMO config file
├── *.net.xml     # Network file
├── *.rou.xml     # Route file
├── *.log         # Log file with SUMO warnings/errors
└── Dataset/      # Folder for the raw input data (not included in the repo due to size)
```

SUMO can write warnings and errors to a custom log file using the `--error-log` option, which in this project is configured as `sumo_error.log` for easier debugging.

This README emphasizes that users need to **manually verify and update paths** in each script before running, without requiring you to change any code. The troubleshooting section addresses common path-related errors they might encounter.

## Credits

This project was developed by:

- [António Santos](https://github.com/totas30)
- [Gonçalo Dias](@Gonto03)
- [Paulo Silva](@WrekingPanda)