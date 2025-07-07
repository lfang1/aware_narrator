# AWARE Narrator

## Overview
AWARE Narrator is a comprehensive Python toolkit that processes sensor data from mobile devices, performs DBSCAN clustering on location data, and generates detailed narrative descriptions of user mobility and activity patterns. The toolkit integrates multiple sensors including location, applications, keyboard input, screen usage, calls, messages, and more. It includes Google Maps API integration for reverse geocoding and uses a configuration file (`config.yaml`) to customize parameters.


## Installation

This project supports both **Conda** and **pip** for managing and installing dependencies. Follow the instructions below to set up your environment.

### Using Conda

1. Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed.
2. Activate your Conda environment (or create one if needed):

   ```bash
   conda create -n my_env python=3.13
   conda activate my_env
   ```
3. Create (or update) your environment from the file:

   ````bash
   # Create a new environment with the name defined in the file:
   conda env create --file environment.yml

   # Or create under a custom name:
   conda env create -n <your_env_name> --file environment.yml

   # To update an existing environment to match environment.yml:
   conda env update -n <your_env_name> --file environment.yml --prune
   ```bash
   conda env update --file environment.yml --prune
   ````

> **Note:** `environment.yml` was generated with:
>
> ```bash
> conda env export --from-history > environment.yml
> ```

### Using pip

1. Ensure you have Python 3.13 installed.
2. (Optional) Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate   # on Linux/macOS
   venv\\Scripts\\activate  # on Windows
   ```
3. Install all pip dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

> **Tip:** To generate `requirements.txt`, run:
>
> ```bash
> pip freeze > requirements.txt
> ```



## Project Structure
The project includes several Python scripts:

- **`aware_narrator.py`** - Main script that processes sensor data and generates narratives
- **`extract_sessions.py`** - Extracts session boundaries from screen on/off events
- **`json2jsonl.py`** - Utility to convert JSON files to JSONL format
- **`split_description.py`** - Splits output narratives by sensor type

## Configuration File (`config.yaml`)
The scripts require a YAML configuration file with the following structure:

```yaml
pid_to_deviceid_map: "resources/pid_deviceid_mapping.csv" # replace by your own pid to device mapping csv file (make sure if contains pid and device_id)

P_ID: "SS002" # replace by your actual participant id
START_TIME: "2023-07-25 00:00:00"
END_TIME: "2023-07-26 00:00:00"
timezone: "Australia/Melbourne"
input_directory: "participant_data/{P_ID}"
session_data_file: "step1_data/{P_ID}/sessions.jsonl"

sensor_integration_time_window: 60 # minutes
gate_time_window: 5 # minutes

sensors:
  - "applications_foreground"
  - "applications_notifications"
  - "battery"
  - "bluetooth"
  - "calls"
  - "installations"
  - "keyboard"
  - "messages"
  - "screen"
  - "wifi"
  - "sensor_wifi"
  - "locations"

# location sampling parameters
frequency_gps: 180 # seconds
frequency_network: 300 # seconds

output_file: "description/{P_ID}_output.txt"
daily_output_dir: "daily_description/{P_ID}"
DISCARD_SYSTEM_UI: true
GOOGLE_MAP_KEY: ""
eps: 0.0000078  # DBSCAN clustering parameter: ~50m radius
min_samples: 5  # DBSCAN clustering parameter: minimum points to form cluster
night_time_start: 22 # Start of nighttime (24-hour format)
night_time_end: 6 # End of nighttime (24-hour format)
```

### Key Configuration Parameters:

#### Data Source Parameters:
- **`pid_to_deviceid_map`**: CSV file mapping participant IDs to device IDs
- **`P_ID`**: Participant ID to process
- **`START_TIME` / `END_TIME`**: Time range for data processing (YYYY-MM-DD HH:MM:SS format)
- **`timezone`**: Timezone for timestamp conversion (e.g., "Australia/Melbourne")
- **`input_directory`**: Path to participant data folder (supports {P_ID} placeholder)
- **`session_data_file`**: Path to sessions.jsonl file for application and keyboard analysis

#### Processing Parameters:
- **`sensor_integration_time_window`**: Time window in minutes for sensor data integration
- **`gate_time_window`**: Time window in minutes for WiFi and Bluetooth scan integration
- **`sensors`**: List of sensors to include in analysis
- **`frequency_gps` / `frequency_network`**: Location sampling rates in seconds

#### Output Parameters:
- **`output_file`**: Path for the main narrative output (supports {P_ID} placeholder)
- **`daily_output_dir`**: Directory for daily output files (supports {P_ID} placeholder)
- **`DISCARD_SYSTEM_UI`**: Whether to filter out system UI applications

#### Location Clustering Parameters:
- **`GOOGLE_MAP_KEY`**: (Optional) API key for Google Maps reverse geocoding
- **`eps`**: DBSCAN epsilon parameter (distance threshold in radians, ~50m = 0.0000078)
- **`min_samples`**: DBSCAN minimum samples to form a cluster
- **`night_time_start` / `night_time_end`**: Hours defining nighttime for home location identification

## How to Run the Scripts

### 1. PID to Device ID Mapping Generator

Generate the participant ID to device ID mapping file required by the main analysis script:

```sh
python map_pid_deviceid.py
```

This script processes a CSV file containing participant information and creates a mapping file used by `aware_narrator.py`.

**Input Requirements:**
- CSV file with at least 2 columns:
  - `pid`: Participant ID
  - `device_id`: Device ID (multiple device IDs can be separated by ";")

**Default behavior:**
- Input file: `resources/participants.csv`
- Output file: `resources/pid_deviceid_mapping.csv`

The output mapping file is referenced in `config.yaml` as `pid_to_deviceid_map` and is required for the main analysis script to function properly.

### 2. JSON to JSONL Converter

Convert JSON files to JSONL format:

```sh
# Convert all JSON files in a folder
python json2jsonl.py /path/to/json/folder

# Specify output folder
python json2jsonl.py /path/to/json/folder -o /path/to/output/folder

# Search recursively in subdirectories
python json2jsonl.py /path/to/json/folder -r -o /path/to/output/folder
```

### 3. Extract Sessions Script

Extract session boundaries from screen data:

```sh
# Process a single participant
python extract_sessions.py --participant SS001

# Process all participants in the input directory
python extract_sessions.py --all

# Custom session threshold (default: 45000ms = 45 seconds)
python extract_sessions.py --participant SS001 --threshold 45000

# Custom input/output directories
python extract_sessions.py --participant SS001 --input-dir custom_data --output-dir custom_output
```

### 4. Main Analysis Script

```sh
python aware_narrator.py
```

This processes all sensor data according to the configuration and generates comprehensive narratives.

### 5. Split Description Script

Split output narratives by sensor type:

```sh
python split_description.py
```

This creates separate files for each sensor type in the `description_split/{PID}/` folder.

## Input Data Structure

The project expects the following directory structure:

```
participant_data/
├── {P_ID}/
│   ├── applications_foreground.jsonl
│   ├── applications_notifications.jsonl
│   ├── battery.jsonl
│   ├── bluetooth.jsonl
│   ├── calls.jsonl
│   ├── installations.jsonl
│   ├── keyboard.jsonl
│   ├── locations.jsonl
│   ├── messages.jsonl
│   ├── screen.jsonl
│   ├── wifi.jsonl
│   └── sensor_wifi.jsonl
step1_data/
├── {P_ID}/
│   └── sessions.jsonl
resources/
├── pid_deviceid_mapping_final.csv
└── app_package_pairs.jsonl
```

## Output

The toolkit generates several types of output:

- **Main narrative file**: Comprehensive narrative saved to `output_file` path
- **Daily narratives**: Separate daily files in `daily_output_dir`
- **Session data**: Screen usage sessions in `step1_data/{P_ID}/sessions.jsonl`
- **Split narratives**: Sensor-specific files in `description_split/{P_ID}/`
- **Reverse geocoding**: Location data with address information (if Google Maps API is enabled)
- **Clustering analysis**: Location clusters with home detection

## Sensor Data Analysis

The toolkit analyzes the following sensor types:

- **Location**: GPS and network-based location with DBSCAN clustering
- **Applications**: Foreground app usage with session correlation
- **Keyboard**: Typing patterns and text input analysis
- **Screen**: Screen on/off events and usage sessions
- **Battery**: Battery level and charging status
- **Calls**: Phone call events and durations
- **Messages**: SMS and messaging activity
- **WiFi/Bluetooth**: Network scanning and connectivity
- **Notifications**: Application notification events

## Troubleshooting

- **JSONDecodeError**: Ensure `config.yaml` is properly formatted YAML (no JSON-style comments)
- **ModuleNotFoundError**: Install dependencies using `pip install -r requirements.txt`
- **Google Maps API errors**: Ensure a valid API key is provided in `GOOGLE_MAP_KEY`
- **File not found errors**: Check that participant data follows the expected directory structure
- **Session data missing**: Run `extract_sessions.py` first to generate session boundaries
- **Empty sensor files**: Verify JSONL files contain valid JSON objects, one per line

## License
This project is for research purposes. Contact the developers for usage permissions.

## Contact
For questions, reach out to the maintainers of the AWARE Narrator project.

