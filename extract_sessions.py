"""
Extract Sessions Script

This script processes screen.jsonl data to extract session boundaries based on screen on/off events.
It creates session data with active periods and saves the results to sessions.jsonl.

Usage:
    Process a single participant:
       python extract_sessions.py --participant SS001

    Process all participants:
       python extract_sessions.py --all

    Custom threshold (default is 45000ms = 45 seconds):
       python extract_sessions.py --participant SS001 --threshold 45000

The script uses the following directory structure by default:
  - Input directory: participant_data
  - Output directory: step1_data
"""

import os
import json
import pandas as pd
import argparse


class DataPaths:
    def __init__(self, base_input="participant_data", base_output="step1_data", participant_prefix="", p_id=None):
        """
        Initialize data paths configuration
        
        Parameters:
            base_input (str): Base directory for input data.
            base_output (str): Base directory for output data.
            participant_prefix (str): Prefix for participant folders.
            p_id (str or None): Participant ID.
        """
        self.base_input = base_input
        self.base_output = base_output
        self.participant_prefix = participant_prefix
        self.p_id = p_id
        
    def get_input_dir(self):
        """Get input directory path"""
        if self.p_id:
            return os.path.join(self.base_input, f"{self.participant_prefix}{self.p_id}")
        return self.base_input
    
    def get_output_dir(self):
        """Get output directory path"""
        if self.p_id:
            return os.path.join(self.base_output, f"{self.participant_prefix}{self.p_id}")
        return self.base_output


def load_jsonl(file_path):
    """
    Load a JSONL file into a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the JSONL file.
    
    Returns:
        pd.DataFrame: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    if os.path.getsize(file_path) == 0:
        raise ValueError(f"File is empty: {file_path}")
    
    return pd.read_json(file_path, lines=True)


def get_sessions(paths, threshold=45000):
    """
    Extract session data from screen on/off events.
    
    Parameters:
        paths (DataPaths): Object holding directory paths.
        threshold (int): Minimum time gap in milliseconds to consider as a new session (default: 45000ms = 45 seconds).
    
    Returns:
        pd.DataFrame: DataFrame containing session information with columns:
            - session_id: Sequential session identifier
            - start_timestamp: Session start time in milliseconds
            - end_timestamp: Session end time in milliseconds
            - active_periods: List of dicts with start/end times for active periods
            - duration_seconds: Total session duration in seconds
            - active_seconds: Total active screen time in seconds
    """
    json_file = os.path.join(paths.get_input_dir(), "screen.jsonl")
    df = load_jsonl(json_file)
    
    print(f"Loaded {len(df)} records from screen.jsonl")
    
    # Convert timestamp from nanoseconds to milliseconds
    df['timestamp'] = (pd.to_numeric(df['timestamp']) / 1_000_000).astype(int)
    
    # Only keep screen on/off events and sort by timestamp
    df = df[df['screen_status'].isin([0, 1])].sort_values('timestamp')
    
    print(f"Found {len(df)} screen on/off events")
    
    sessions = []
    session_start = None
    last_timestamp = None
    active_period_start = None
    active_periods = []
    active_seconds = 0
    
    for i in range(len(df)):
        current_status = df.iloc[i]['screen_status']
        current_timestamp = df.iloc[i]['timestamp']

        if current_status == 1:  # Screen turning on
            active_period_start = current_timestamp
            if last_timestamp is None or (current_timestamp - last_timestamp > threshold):
                if session_start is not None:
                    sessions.append({
                        'session_id': len(sessions) + 1,
                        'start_timestamp': int(session_start),
                        'end_timestamp': int(last_timestamp),
                        'active_periods': active_periods,
                        'duration_seconds': round((last_timestamp - session_start) / 1000, 2),
                        'active_seconds': round(active_seconds / 1000, 2)
                    })
                session_start = current_timestamp
                active_periods = []
                active_seconds = 0
                
        elif current_status == 0 and active_period_start is not None:
            period_duration = current_timestamp - active_period_start
            active_seconds += period_duration
            active_periods.append({
                'start': int(active_period_start),
                'end': int(current_timestamp)
            })
            active_period_start = None
            
        last_timestamp = current_timestamp
    
    # Add final session if present
    if session_start is not None:
        sessions.append({
            'session_id': len(sessions) + 1,
            'start_timestamp': int(session_start),
            'end_timestamp': int(last_timestamp),
            'active_periods': active_periods,
            'duration_seconds': round((last_timestamp - session_start) / 1000, 2),
            'active_seconds': round(active_seconds / 1000, 2)
        })
    
    result_df = pd.DataFrame(sessions)
    
    output_dir = paths.get_output_dir()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir, "sessions.jsonl")
    result_df.to_json(output_file, orient='records', lines=True)
    
    print(f"Extracted {len(sessions)} sessions")
    print(f"Sessions saved to: {output_file}")
    
    return result_df


def extract_sessions_for_participant(participant_id, base_input="participant_data", base_output="step1_data", threshold=45000):
    """
    Extract sessions for a single participant.
    
    Parameters:
        participant_id (str): Participant ID to process.
        base_input (str): Base directory for input data.
        base_output (str): Base directory for output data.
        threshold (int): Session threshold in milliseconds.
    
    Returns:
        pd.DataFrame: Session data for the participant.
    """
    paths = DataPaths(base_input=base_input, base_output=base_output, p_id=participant_id)
    return get_sessions(paths, threshold=threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Extract session data from screen.jsonl files for participants."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p', '--participant', type=str, help="Participant ID to process (e.g., 1234)")
    group.add_argument('--all', action='store_true', help="Process all participants")
    parser.add_argument('--threshold', type=int, default=45000, 
                        help="Session threshold in milliseconds (default: 45000ms = 45 seconds)")
    parser.add_argument('--input-dir', type=str, default="participant_data",
                        help="Base input directory (default: participant_data)")
    parser.add_argument('--output-dir', type=str, default="step1_data",
                        help="Base output directory (default: step1_data)")
    
    args = parser.parse_args()

    if args.participant:
        print(f"Extracting sessions for participant {args.participant}...")
        print(f"Using threshold: {args.threshold}ms ({args.threshold/1000} seconds)")
        
        try:
            sessions_df = extract_sessions_for_participant(
                args.participant, 
                base_input=args.input_dir,
                base_output=args.output_dir,
                threshold=args.threshold
            )
            print(f"Successfully processed participant {args.participant}")
            print(f"Total sessions found: {len(sessions_df)}")
            if len(sessions_df) > 0:
                total_duration = sessions_df['duration_seconds'].sum()
                total_active = sessions_df['active_seconds'].sum()
                print(f"Total session duration: {total_duration:.2f} seconds ({total_duration/3600:.2f} hours)")
                print(f"Total active time: {total_active:.2f} seconds ({total_active/3600:.2f} hours)")
        except Exception as e:
            print(f"Error processing participant {args.participant}: {str(e)}")
    else:
        print("Extracting sessions for all participants...")
        print(f"Using threshold: {args.threshold}ms ({args.threshold/1000} seconds)")
        
        participant_dirs = [d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))]
        if not participant_dirs:
            print(f"No participant directories found in {args.input_dir}")
            return
            
        successful = 0
        failed = 0
        
        for participant in participant_dirs:
            print(f"\nProcessing participant {participant}...")
            try:
                sessions_df = extract_sessions_for_participant(
                    participant,
                    base_input=args.input_dir,
                    base_output=args.output_dir,
                    threshold=args.threshold
                )
                print(f"Successfully processed participant {participant} - {len(sessions_df)} sessions found")
                successful += 1
            except Exception as e:
                print(f"Error processing participant {participant}: {str(e)}")
                failed += 1
        
        print(f"\nProcessing complete:")
        print(f"Successfully processed: {successful} participants")
        print(f"Failed: {failed} participants")


if __name__ == "__main__":
    main() 