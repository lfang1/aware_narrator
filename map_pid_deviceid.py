import csv
import os

def generate_pid_deviceid_mapping(input_file='resources/participants.csv', output_file='resources/pid_deviceid_mapping.csv'):
    """
    Read participants.csv and generate a subset CSV with PID mapping to device_id.
    Multiple device_ids are kept in one row, separated by ";" in the device_id column.
    
    Args:
        input_file (str): Path to the input participants.csv file
        output_file (str): Path to the output mapping CSV file
    """
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return
    
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            
            # Read the header
            header = next(reader)
            
            # Find the indices of device_id and pid columns
            try:
                device_id_index = header.index('device_id')
                pid_index = header.index('pid')
            except ValueError as e:
                print(f"Error: Required column not found in CSV header: {e}")
                return
            
            # Prepare data for output
            mapping_data = []
            
            # Process each row
            for row_num, row in enumerate(reader, start=2):  # start=2 because header is row 1
                if len(row) <= max(device_id_index, pid_index):
                    print(f"Warning: Row {row_num} has insufficient columns, skipping.")
                    continue
                
                device_id = row[device_id_index].strip()
                pid = row[pid_index].strip()
                
                if not device_id or not pid:
                    print(f"Warning: Row {row_num} has empty device_id or pid, skipping.")
                    continue
                
                # Keep multiple device IDs in one row, separated by semicolons
                mapping_data.append([pid, device_id])
        
        # Write the output CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            
            # Write header
            writer.writerow(['pid', 'device_id'])
            
            # Write data
            writer.writerows(mapping_data)
        
        print(f"Successfully generated mapping file: '{output_file}'")
        print(f"Total participants: {len(mapping_data)}")
        
        # Display first few mappings as preview
        if mapping_data:
            print("\nPreview (first 10 mappings):")
            print("PID\t\tDevice ID(s)")
            print("-" * 80)
            for i, (pid, device_id) in enumerate(mapping_data[:10]):
                # Truncate long device ID strings for display
                display_devices = device_id if len(device_id) <= 60 else device_id[:57] + "..."
                print(f"{pid}\t\t{display_devices}")
            if len(mapping_data) > 10:
                print("...")
                
    except Exception as e:
        print(f"Error processing file: {e}")

def main():
    """Main function to run the script"""
    print("PID to Device ID Mapping Generator")
    print("(Multiple device IDs per row, separated by semicolons)")
    print("=" * 60)
    
    # Generate the mapping
    generate_pid_deviceid_mapping()

if __name__ == "__main__":
    main()
