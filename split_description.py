#!/usr/bin/env python3
"""
Script to split sensor data by sensor type from SS010_output.txt
Each sensor type gets its own file in the splitted_output folder
"""

import os
import re
from collections import defaultdict

def split_sensor_data(input_file, output_folder):
    """
    Split sensor data by sensor type and save to separate files
    
    Args:
        input_file (str): Path to the input file
        output_folder (str): Path to the output folder
    """
    
    # Extract PID from input filename (assuming format: {pid}_output.txt)
    input_basename = os.path.basename(input_file)
    if '_output.txt' in input_basename:
        pid = input_basename.replace('_output.txt', '')
    else:
        # Fallback: use filename without extension as PID
        pid = os.path.splitext(input_basename)[0]
    
    # Create output folder with PID subfolder
    pid_output_folder = os.path.join(output_folder, pid)
    os.makedirs(pid_output_folder, exist_ok=True)
    
    # Dictionary to store lines for each sensor type
    sensor_data = defaultdict(list)
    
    # Regular expression to match timestamp pattern (YYYY-MM-DD HH:MM:SS)
    timestamp_pattern = re.compile(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
    
    # Variables to track current sensor entry
    current_sensor_type = None
    current_line_parts = []
    
    # Read the input file and group lines by sensor type
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n\r')  # Remove newline but keep other whitespace
                
                if not line.strip():  # Skip empty lines
                    continue
                
                # Check if this line starts with a timestamp
                if timestamp_pattern.match(line):
                    # This is a new sensor entry
                    # First, save the previous entry if it exists
                    if current_sensor_type and current_line_parts:
                        complete_line = '\n'.join(current_line_parts)
                        sensor_data[current_sensor_type].append(complete_line)
                    
                    # Parse the new line
                    parts = line.split(' | ')
                    if len(parts) >= 3:
                        current_sensor_type = parts[1].strip()
                        current_line_parts = [line]
                    else:
                        print(f"Warning: Line {line_num} has unexpected format: {line}")
                        current_sensor_type = None
                        current_line_parts = []
                else:
                    # This is a continuation line
                    if current_sensor_type and current_line_parts:
                        current_line_parts.append(line)
                    else:
                        print(f"Warning: Line {line_num} appears to be a continuation but no current sensor entry: {line}")
            
            # Don't forget to save the last entry
            if current_sensor_type and current_line_parts:
                complete_line = '\n'.join(current_line_parts)
                sensor_data[current_sensor_type].append(complete_line)
                
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    # Write each sensor's data to a separate file
    for sensor_type, lines in sensor_data.items():
        # Create a safe filename by replacing spaces and special characters
        safe_filename = sensor_type.replace(' ', '_').replace('/', '_')
        output_file = os.path.join(pid_output_folder, f"{safe_filename}.txt")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for line in lines:
                    f.write(line + '\n')
            
            print(f"Created {output_file} with {len(lines)} entries")
        except Exception as e:
            print(f"Error writing to {output_file}: {e}")

def main():
    input_file = "description/SS002_output.txt"
    output_folder = "description_split"
    
    # Extract PID for display purposes
    input_basename = os.path.basename(input_file)
    if '_output.txt' in input_basename:
        pid = input_basename.replace('_output.txt', '')
    else:
        pid = os.path.splitext(input_basename)[0]
    
    print(f"Splitting sensor data from {input_file}")
    print(f"Output will be saved to {output_folder}/{pid}/")
    print("-" * 50)
    
    split_sensor_data(input_file, output_folder)
    
    print("-" * 50)
    print("Splitting complete!")

if __name__ == "__main__":
    main()
