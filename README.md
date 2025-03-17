# AWARE Narrator

## Overview
AWARE Narrator is a Python script that processes sensor data, performs hierarchical clustering on location data, and generates a narrative description of user mobility patterns. It integrates Google Maps API for reverse geocoding and uses a configuration file (`config.yaml`) to customize parameters.

## Prerequisites
Before running the script, ensure you have the following installed:

- Python 3.7+
- Required Python libraries:
  ```sh
  pip install pandas numpy scipy scikit-learn pyyaml googlemaps geopy matplotlib astropy pytz
  ```

## Configuration File (`config.yaml`)
The script requires a YAML configuration file with the following structure:

```yaml
DEVICE_IDs:
  - "c7c7435a-f2fc-41be-9a50-39f2f7b035ed"
  - "017efbf0-cbc0-4441-9e9a-6d3e61a3e389"
SS_ID: "USER_ID"
START_TIME: "1994-09-14 00:00:00" 
END_TIME: "1994-09-20 23:59:59"
timezone: "Australia/Victoria"
csv_directory: "./input_dir"
sensors:
  - "applications_foreground"
  - "applications_notifications"
  - "battery"
  - "bluetooth"
  - "calls"
output_file: "./output_file.txt"
daily_output_dir: "./output_dir"
DISCARD_SYSTEM_UI: true
GOOGLE_MAP_KEY: "YOUR_GOOGLE_MAPS_API_KEY"
hierarchy_linkage: "complete"
night_time_start: 20
night_time_end: 4
```

### Key Configuration Parameters:
- `DEVICE_IDs`: List of device IDs to process.
- `START_TIME` / `END_TIME`: Time range for data processing.
- `timezone`: Timezone for timestamp conversion.
- `csv_directory`: Path to the folder containing CSV sensor data.
- `sensors`: List of sensors to be included in the analysis.
- `output_file`: Path for the final narrative output.
- `GOOGLE_MAP_KEY`: (Optional) API key for Google Maps reverse geocoding.
- `hierarchy_linkage`: Clustering method (`complete` for longest distance, `single` for shortest).
- `night_time_start` / `night_time_end`: Define nighttime hours for home location identification.

## How to Run the Script

1. **Ensure dependencies are installed** (see prerequisites).
2. **Prepare the `config.yaml` file** with the desired parameters.
3. **Run the script**:
   ```sh
   python AWARE_Narrator.py
   ```

## Output
- A **processed narrative** summarizing the movement patterns is saved in the `output_file` path.
- Clustered location data is analyzed and displayed.

## Troubleshooting
- **JSONDecodeError**: Ensure `config.yaml` is properly formatted and doesn’t contain invalid JSON-style comments (`//`).
- **ModuleNotFoundError**: Reinstall dependencies using `pip install -r requirements.txt`.
- **Google Maps API errors**: Ensure a valid API key is provided in `GOOGLE_MAP_KEY`.

## License
This project is for research purposes. Contact the developers for usage permissions.

## Contact
For questions, reach out to the maintainers of the AWARE Narrator project.

