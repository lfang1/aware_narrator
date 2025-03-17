import os
import pandas as pd
import pytz
import json
from datetime import datetime
import pprint
import pytz
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import csv
import os
from astropy import units as u
import re
from datetime import datetime
from collections import defaultdict
import numpy as np
import googlemaps
import numpy as np
from datetime import datetime
from geopy.distance import geodesic
from sklearn.cluster import AgglomerativeClustering
import yaml

# Load configuration from JSON file
CONFIG_FILE = "./config.yaml"
with open(CONFIG_FILE, "r") as file:
    CONFIG = yaml.safe_load(file)

# Assign variables from JSON
DEVICE_IDs = CONFIG["DEVICE_IDs"]
SS_ID = CONFIG["SS_ID"]
START_TIME = CONFIG["START_TIME"]
END_TIME = CONFIG["END_TIME"]
timezone = pytz.timezone(CONFIG["timezone"])
csv_directory = CONFIG["csv_directory"]
sensors = CONFIG["sensors"]
GOOGLE_MAP_KEY = CONFIG["GOOGLE_MAP_KEY"]
output_file = CONFIG["output_file"]
daily_output_dir = CONFIG["daily_output_dir"]
DISCARD_SYSTEM_UI = CONFIG["DISCARD_SYSTEM_UI"]
hierarchy_linkage = CONFIG["hierarchy_linkage"]
night_time_start = CONFIG["night_time_start"]
night_time_end = CONFIG["night_time_end"]

# Ensure output directory exists
os.makedirs(daily_output_dir, exist_ok=True)

csv_files = [(f"phone_{sensor}_raw.csv", sensor) for sensor in sensors]



######################################### Define important locations as needed #########################################
# You can design the map areas and common locations here. This is an example map coordinates of University of Melbourne and example common locations in Melbourne.
MAP = [
    #original
    ("Veterinary preclinical sciences", [(144.95379, -37.79828), (144.95435, -37.79783), (144.95358, -37.79729), (144.95300, -37.79762)]),
    ("Ruth Bishop building, Elizabeth Blackburn school of science, Nancy Millis building, Bio21 business incubator building", [(144.95360, -37.79750), (144.95458, -37.79762), (144.95474, -37.79691), (144.95381, -37.79682)]),
    ("main campus", [(144.95842, -37.79537), (144.96474, -37.79586), (144.96357, -37.80245), (144.95750, -37.80181)]),
    ("physics building side (east of main campus)", [(144.96468, -37.79690), (144.96599, -37.79705), (144.96574, -37.79843), (144.96447, -37.79826)]),
    ("Melbourne Connect", [(144.96435, -37.79916), (144.96557, -37.79930), (144.96535, -37.80054), (144.96409, -37.80036)]),
    ("Law", [(144.95966, -37.80211), (144.96055, -37.80220), (144.96040, -37.80270), (144.95969, -37.80262)]),
    ("Kwong Lee Dow Building", [(144.96044, -37.80415), (144.96108, -37.80424), (144.96122, -37.80366), (144.96057, -37.80354)]),
    ("11 Barry st (west of Kwong Lee Dow building)", [(144.96013, -37.80412), (144.95978, -37.80337), (144.95924, -37.80348), (144.95961, -37.80419)]),
    ("Little Hall", [(144.96278, -37.80323), (144.96340, -37.80327), (144.96328, -37.80390), (144.96266, -37.80380)]),
    ("Audiology clinic", [(144.96357, -37.80372), (144.96416, -37.80376), (144.96411, -37.80398), (144.96354, -37.80392)]),
    ("Southbank", [(144.96743, -37.82367), (144.96980, -37.82356), (144.96676, -37.82606), (144.97058, -37.82498)]),

    #unimelb
    ("Stop1", [(144.963714, -37.799565), (144.963935, -37.799588), (144.963824, -37.800213), (144.963601, -37.800183)]),
    ("Alan Gilbert Building", [(144.958916, -37.799994), (144.959665, -37.800086), (144.959630, -37.800401), (144.958878, -37.800305)]),
    ("ERC Library", [(144.96258, -37.79923), (144.96309, -37.79929), (144.96305, -37.79952), (144.962543, -37.799432)]),
    ("School of Chemistry", [(144.961810, -37.797653), (144.962691, -37.797766), (144.962600, -37.798293), (144.961742, -37.798170)]),
    ("Bailie Library", [(144.959114, -37.797956), (144.959595, -37.797933), (144.959521, -37.798749), (144.959053, -37.798721)]),
    ("Arts West Building", [(144.959161, -37.797437), (144.959670, -37.797504), (144.959595, -37.797933), (144.959114, -37.797956)]),
    ("Old Arts Building", [(144.959886, -37.797370), (144.960379, -37.797445), (144.960360, -37.798138), (144.959780, -37.798075)]),
    ("Redmond Barry Building", [(144.962394, -37.796626), (144.963098, -37.796701), (144.963071, -37.796895), (144.962377, -37.796821)]),
    ("Old Engineering Building", [(144.961374, -37.799212), (144.962085, -37.799290), (144.962036, -37.799796), (144.961301, -37.799671)]),
    ("Glyn Davis Building", [(144.962380, -37.796846), (144.963222, -37.796947), (144.963140, -37.797506), (144.962275, -37.797408)]),
    ("North Lawn", [(144.961531, -37.796713), (144.962354, -37.796832), (144.962239, -37.797431), (144.961431, -37.797298)]),
    ("Student Pavilion", [(144.963376, -37.798504), (144.963820, -37.798546), (144.963784, -37.798770), (144.963337, -37.798718)]),
    ("Tennis Courts", [(144.962568, -37.794961), (144.962885, -37.795002), (144.962787, -37.795602), (144.962453, -37.795561)]),
    ("Athletic Field", [(144.960390, -37.795030), (144.962379, -37.795250), (144.962243, -37.795869), (144.960333, 37.795757)]),

    #city
    ("Queen Victoria Market", [(144.956092, -37.805958), (144.958110, -37.806208), (144.957783, -37.808026), (144.955823, -37.807784)]),
    ("Queen Victoria Market Parking", [(144.955823, -37.807784), (144.957783, -37.808026), (144.957534, -37.808937), (144.955657, -37.808726)]),
    ("Fitzroy Gardens", [(144.978402, -37.809943), (144.983337, -37.810481), (144.982470, -37.815938), (144.977501, -37.815416)]),
    ("State Library Victoria", [(144.964389, -37.809538), (144.966226, -37.808993), (144.966620, -37.809857), (144.964756, -37.810392)]),
    ("QV Melbourne", [(144.964463, -37.810556), (144.966606, -37.809951), (144.966977, -37.810738), (144.964855, -37.811366)]),
    ("Flagstaff Gardens", [(144.952720, -37.809312), (144.954907, -37.808666), (144.956141, -37.8116209), (144.954049, -37.812231)]),
    ("Melbourne Zoo", [(144.948697, -37.783345), (144.951947, -37.781903), (144.954695, -37.784027), (144.950707, -37.787424)]),
    ("Melbourne Museum", [(144.970142, -37.802966), (144.971878, -37.802447), (144.973121, -37.803270), (144.972955, -37.803893), (144.970185, -37.803575)]),
    ("Royal Exhibition Building", [(144.970714, -37.804078), (144.972452, -37.804285), (144.972378, -37.805169), (144.970531, -37.804985)]),
    ("Carlton Gardens", [(144.969456, -37.805030), (144.973378, -37.805480), (144.973026, -37.807536), (144.969088, -37.807121)]),
    ("Melbourne Central", [(144.96158, -37.810376), (144.963741, -37.809771), (144.963874, -37.810084), (144.963422, -37.810220), (144.963593, -37.810719), (144.963389, -37.810878), (144.963818, -37.811657), (144.963237, -37.811829), (144.962889, -37.811010), (144.961958, -37.8111417)]),
    ("Emporium Melbourne", [(144.96312, -37.812177), (144.964290, -37.811814), (144.964662, -37.812575), (144.963401, -37.812948)]),
    ("National Gallery of Victoria", [(144.968120, -37.822167), (144.969129, -37.821883), (144.969735, -37.823154), (144.968744, -37.823504), (144.967637, -37.823268)]),
    ("Flinders Street Station", [(144.963523, -37.818702), (144.967307, -37.817646), (144.967817, -37.818719), (144.964241, -37.819187)])
]


common_addresses = (
            "Carlton VIC 3053", "Carlton North VIC 3054", "Fitzroy VIC 3065", "Fitzroy North VIC 3068",
            "Parkville VIC 3052", "Port Melbourne VIC 3207", "Melbourne VIC 3000", "Melbourne VIC 3004",
            "North Melbourne VIC 3051", "East Melbourne VIC 3002", "South Melbourne VIC 3205", "West Melbourne VIC 3003"
        )





def structure_time(dt_object):
    """Formats a datetime object into a structured string format."""
    return dt_object.strftime("%a %b %d %H:%M:%S")

def generate_description_applications_foreground(new_result, sensor, dt_object):
    """Generates a description for foreground applications detected."""
    if sensor == "applications_foreground":
        app_name = new_result.get("application_name", "Unknown")
        if app_name != "System UI":
            return structure_time(dt_object) + " | applications | Opened the app " + app_name
    return None


def generate_description_applications_notifications(new_result, sensor, dt_object):    
    """Generates a description for application notifications received."""
    if sensor == "applications_notifications":
        if "text" in new_result and new_result["text"] and new_result["text"].startswith("[") and new_result["text"] !="[]":
            return structure_time(dt_object) + " | notifications | Received a notification from the " + new_result["application_name"] + ". The content of the notification was " + new_result["text"] 
        elif "application_name" in new_result and new_result["application_name"] is not None:
            return structure_time(dt_object) + " | notifications | Received a notification from the " + new_result["application_name"] 
        else:
            return structure_time(dt_object) + " | notifications | Received a notification from the " + new_result.get("application_name", "Unknown")


def generate_description_battery(new_result, sensor, dt_object):    
    """Generates a description for battery-related events."""
    if sensor == "battery": 
        statuses = {
            -2: "The phone rebooted",
            -1: "The phone shutdown", #ignore no.1   
            2: "The phone started charging",
            3: "The phone started discharging",
            4: "The phone was not charging",
            5: "The phone battery became fully charged"
        }
        if "battery_status" in new_result and new_result["battery_status"] in statuses: 
            # Check if 'battery_status' key exists in new_result
            status = statuses.get(new_result.get("battery_status"))
            return structure_time(dt_object) + f" | battery | {status}, the battery level was {new_result.get('battery_level', 'Unknown')}"
    return None
        

def generate_description_bluetooth(new_result, sensor, dt_object):   
    """Generates a description for detected nearby Bluetooth devices."""     
    if sensor == "bluetooth":
        if "bt_name" in new_result and new_result["bt_name"] != "":
            return structure_time(dt_object) + ' | bluetooth | Detected the nearby bluetooth device  "' + new_result["bt_name"] + '"'
        else:
            return structure_time(dt_object) + " | bluetooth | Detected a nearby bluetooth device"


def generate_description_sensor_bluetooth(new_result, sensor, dt_object):
    """Generates a description for connected Bluetooth devices."""
    if sensor == "sensor_bluetooth":
        if "bt_name" in new_result and new_result["bt_name"] != "":
            return structure_time(dt_object) + ' | bluetooth | Connected to the bluetooth device "' + new_result["bt_name"] + '"'
        else:
            return structure_time(dt_object) + " | bluetooth | Connected to a bluetooth device"


def generate_description_calls(new_result, sensor, dt_object):    
    """Generates a description for phone call activities."""  
    if sensor == "calls":
        if "trace" in new_result:
            trace = new_result["trace"]
        else:
            trace = None
        message_number = get_message_number(trace) 
        calls_type = {
            1: "Received a phone call from person ",
            2: "Made a phone call to person ",
            3: "Missed a call from person "
        }
        if "call_type" in new_result and new_result["call_type"] in calls_type:
            call_description = calls_type.get(new_result.get("call_type"))
            if "call_duration" in new_result:
                return structure_time(dt_object) + " | calls | " + call_description + str(message_number)+ ". The call lasted " + str(new_result["call_duration"]) + " seconds"
            else:
                return structure_time(dt_object)  + " | calls | " + call_description + str(message_number)


def generate_description_installations(new_result, sensor, dt_object):
    """Generates a description for app installation activities."""
    if sensor == "installations":
        statuses = {
            0: "was removed",
            1: "was added",
            2: "was updated"
        }
        status = statuses.get(new_result.get("installation_status", "Unknown"), "Unknown")
        return structure_time(dt_object) + f" | installations | {new_result.get('application_name', 'Unknown')} {status}"


def generate_description_keyboard(new_result, sensor, dt_object):
    """Generates a description for keyboard input activities."""
    if sensor == "keyboard":
        if new_result["current_text"]:
            cleaned_text = new_result.get("current_text", "").strip("[]")
            if cleaned_text:
                return structure_time(dt_object) + " | keyboard | Entered the following text into the phone keyboard: " + cleaned_text         

def generate_description_messages(new_result, sensor, dt_object):
    """Generates a description for messaging activities."""
    if sensor == "messages":
        trace = new_result["trace"]
        message_number = get_message_number(trace) 
        if new_result["message_type"] == 1:
            return structure_time(dt_object) + " | messages | Received a message from person" + str(message_number)
        elif new_result["message_type"] == 2:
            return structure_time(dt_object) + " | messages | Sent a message to person" + str(message_number)

        
def generate_description_screen(new_result, sensor, dt_object):
    """Generates a description for screen status changes."""
    if sensor == "screen" : 
        screen_statuses ={
            0:"Phone screen turned off",
            1:"Phone screen turned on",
            2:"Phone screen locked",
            3:"Phone screen unlocked"
        }
        if "screen_status" in new_result and new_result["screen_status"] in screen_statuses: 
            status = screen_statuses.get(new_result.get("screen_status"))
            return structure_time(dt_object) + " | screen status | " + status
        else:
            return None

def generate_description_touch(new_result, sensor, dt_object):
    """Generates a description for touch interactions on the screen."""
    if sensor == "touch":
        t_text = ""
        t_app = ""
        actions = {
            "ACTION_AWARE_TOUCH_CLICKED": "Clicked",
            "ACTION_AWARE_TOUCH_LONG_CLICKED": "Clicked longer",
            "ACTION_AWARE_TOUCH_SCROLLED_DOWN": "Scrolled down within a view",
            "ACTION_AWARE_TOUCH_SCROLLED_UP": "Scrolled up within a view"
        }
        if new_result["touch_action"] in actions:
            action = actions.get(new_result.get("touch_action"))
        if new_result["touch_action_text"] != "":
            t_text = new_result.get("touch_action_text").strip("[]")
            if t_text != "":
                t_text = f' on "{t_text}"'
        if new_result["touch_app"] in application_name_list:
            t_app = application_name_list.get(new_result["touch_app"])
        return structure_time(dt_object) + f' | touch | {action}{t_text} in the app "{t_app}".'

def generate_description_wifi(new_result, sensor, dt_object):
    """Generates a description for detected Wi-Fi networks."""
    if sensor == "wifi":
        if new_result["ssid"] != "":
            return structure_time(dt_object) + ' | wifi | Detected the nearby wifi network "' + new_result["ssid"] + '"'
        else:
            return structure_time(dt_object) + " | wifi | Detected a nearby wifi network"

  
   
def generate_description_sensor_wifi(new_result, sensor, dt_object, prev_sensor_wifi, narrative_list):
    """Generates a description for connected Wi-Fi networks."""
    description = None
    lst = []
    if sensor == "sensor_wifi":
        if new_result["ssid"] != "":
            description = structure_time(dt_object) + ' | wifi | Connected to the wifi network ' + new_result["ssid"] 
        else:
            description = structure_time(dt_object) + " | wifi | Connected to a wifi network"
        
    if description:
        current_wifi_name = new_result.get("ssid")
        if prev_sensor_wifi is None or prev_sensor_wifi.get("ssid") != current_wifi_name:
            narrative_list.append((dt_object, description))
        prev_sensor_wifi = current_wifi_name #save current result
    return narrative_list
        

def haversine(lat1, lon1, lat2, lon2):
    """Computes the Haversine distance between two geographic points."""
    R = 6371000 * u.m #diameter of the earth (m)

    #convert latitude and longitude to radian (float)
    phi1 = np.radians(float(lat1))
    phi2 = np.radians(float(lat2))
    delta_phi = np.radians(float(lat2) - float(lat1))
    delta_lambda = np.radians(float(lon2) - float(lon1))
    
    a = np.sin(delta_phi / 2.0) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c).to(u.m)


def is_point_in_polygon(lat, lon, poly):
    """
    Check if each point is included in a polygon.

    :param lat: Latitude of the point.
    :param lon: Longitude of the point.
    :param poly: List of vertices of the polygon [(lat1, lon1), (lat2, lon2), ...].
    :return: True (included), False (not included).
    """
    n = len(poly)
    inside = False
    x, y = lon, lat
    p1x, p1y = poly[0]
    for i in range(n + 1):  # Loop condition +1 to return to the first point
        p2x, p2y = poly[i % n]
        if lat > min(p1y, p2y):
            if lat <= max(p1y, p2y):
                if lon <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lon <= xinters:
                        inside = not inside  # Determine based on whether the number of intersections is odd or even
        p1x, p1y = p2x, p2y
    return inside


def is_point_in_any_polygon(lat, lon, areas):
    """
    check if each point is included in polygons
    :param lat: latitude of the point
    :param lon: longitude of the point
    :param areas: list of polygon [{name, coordinates, contains}]
    :return: list of included places
    """
    included_places = []
    for area in areas:
        if area["contains"](lat, lon):
            included_places.append(area["name"])
    return included_places


# Dictionary to store message traces and keep track of unique IDs
message_traces = {}
message_trace_n = 0
def get_message_number(trace):
    """Assigns a unique number to each message trace."""
    global message_trace_n
    if trace not in message_traces:
        message_trace_n += 1
        message_traces[trace] = message_trace_n
    return message_traces[trace]


def generate_description_locations(lst, all_points, cluster_labels, cluster):
    """
    Generates descriptions of user locations based on clustering results.
    
    :param lst: List to store previous descriptions.
    :param all_points: List of all location points.
    :param cluster_labels: Labels assigned to each point by clustering.
    :param cluster: Cluster information containing place names and distances.
    :return: Updated list with new location descriptions.
    """
    for i in range(len(all_points)):
        record_time = all_points[i][2]
        label_now = cluster_labels[i]
        place_now = cluster[label_now][4]

        #place and distance
        if place_now == "home":
            distance_from_home_now = None
        else:
            distance_from_home_now = f"{cluster[label_now][5]:.1f}m from home"

        #convert speed to activity
        if all_points[i][3] == 0:
            speed_now = "stopping"

        elif 0 < all_points[i][3] <= 1:
            speed_now = "walking"
        elif 1 < all_points[i][3] <= 3:
            speed_now = "running"
        elif all_points[i][3] > 3:
            speed_now = "riding vehicle"

        if distance_from_home_now: 
            description = f"{structure_time(record_time)} | locations | {place_now}, {distance_from_home_now}, {speed_now}"
        else:
            description = f"{structure_time(record_time)} | locations | {place_now}, {speed_now}"
        lst.append((all_points[i][2], description))
    return lst

def compute_distance_matrix_chunked(all_points, chunk_size=1000):
    """
    Computes a distance matrix using the Haversine formula in chunks to handle large datasets efficiently.
    
    :param all_points: List of location data points.
    :param chunk_size: Number of points processed per iteration to save memory.
    :return: A symmetric distance matrix with distances between all points.
    """
    num_points = len(all_points)
    distance_matrix = np.zeros((num_points, num_points))
    
    for start in range(0, num_points, chunk_size):
        print(f"processing row: {start} to {start + 1000}")
        end = min(start + chunk_size, num_points)
        for i in range(start, end):
            lat1, lon1 = all_points[i, 0], all_points[i, 1]
            for j in range(i + 1, num_points):
                lat2, lon2 = all_points[j, 0], all_points[j, 1]
                distance = haversine(lat1, lon1, lat2, lon2)
                distance = distance.to_value() if isinstance(distance, u.Quantity) else distance
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
    
    return distance_matrix


def read_file(file_path):
    """Reads a file and returns its content as a list of lines."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def process_lines(lines):
    """
    Processes log file lines into structured log entries.
    
    :param lines: List of lines from the log file.
    :return: List of structured log entries.
    """
    log_entries = []
    current_entry = ""
    
    for line in lines:
        if re.match(r'^[A-Z][a-z]{2} [A-Z][a-z]{2} \d{2} \d{2}:\d{2}:\d{2}', line):
            if current_entry:
                log_entries.append(current_entry.strip())
            current_entry = line
        else:
            current_entry += " " + line.strip()
    
    if current_entry:
        log_entries.append(current_entry.strip())
    
    return log_entries

def split_entries_by_date(log_entries, default_year):
    """
    Splits log entries by their respective dates.
    
    :param log_entries: List of structured log entries.
    :param default_year: Default year to append to date entries.
    :return: Dictionary mapping dates to log entries.
    """
    date_entries = defaultdict(list)
    
    for entry in log_entries:
        date_str = entry.split(' | ')[0]
        date = datetime.strptime(date_str + f" {default_year}", '%a %b %d %H:%M:%S %Y').date()
        date_entries[date].append(entry)
    
    return date_entries

def write_entries_to_files(date_entries, output_dir):
    """Writes log entries into separate files based on their date."""
    for date, entries in date_entries.items():
        file_name = f"{output_dir}/{date}.txt"
        with open(file_name, 'w') as file:
            for entry in entries:
                file.write(entry + '\n')

def week_to_day(input_file, output_dir):
    """Converts weekly log data into daily log files."""
    lines = read_file(input_file)
    log_entries = process_lines(lines)
    date_entries = split_entries_by_date(log_entries, 2023)
    print(date_entries)
    write_entries_to_files(date_entries, output_dir)
    
    
########################################## Description generation #########################################

# Convert start and end time strings into timezone-aware datetime objects
from_date = datetime.strptime(START_TIME,"%Y-%m-%d %H:%M:%S") 
from_date = timezone.localize(from_date)
to_date = datetime.strptime(END_TIME, "%Y-%m-%d %H:%M:%S") 
to_date = timezone.localize(to_date)
new_results = []
key_list = ["battery_status", "battery_level", "call_type","call_duration", "installation_status", "message_type", "screen_status"]

print("Apply target time limitation to data...")
for csv_file, sensor_name in csv_files:
    file_path = os.path.join(csv_directory, csv_file)
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row['device_id'] in DEVICE_IDs:
                new_result = {}
                for key, value in row.items():
                    if key == "timestamp": # Convert timestamp to float for all sensors
                        value = float(value)
                    elif key in key_list: # Convert categorical key values to integers
                        value = int(value)
                    new_result[key] = value
                new_result['sensor'] = sensor_name
                dt_object = datetime.fromtimestamp(int(new_result["timestamp"]) / 1000, timezone)
                # Apply time filtering
                if from_date <= dt_object <= to_date:
                    new_results.append(new_result)

                    
# Create a dictionary to store app names  
print("Applications preparation: getting application names...")
application_name_list = {}

for new_result in new_results:
    sensor = new_result["sensor"] 
    if sensor == "applications_foreground":
        package_name = new_result.get("package_name")
        application_name = new_result.get("application_name")
        if package_name not in application_name_list:
            application_name_list[package_name] = application_name

# Generate descriptions for most sensors except location
print("Generating descriptions for the majority of sensors (other than locations)")
narrative_list = []
prev_battery_status = None
prev_bluetooth_name = None
prev_keyboard = None
prev_sensor_wifi = None

for new_result in new_results:
    dt_object = datetime.fromtimestamp(int(new_result["timestamp"]) / 1000, timezone)
    sensor = new_result["sensor"]

    # Generate sensor-specific descriptions
    if sensor == "applications_foreground":
        description = generate_description_applications_foreground(new_result, sensor, dt_object)
        if description:
            narrative_list.append((dt_object, description))
    elif sensor == "applications_notifications": 
        description = generate_description_applications_notifications(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
        
    elif sensor == "battery": 
        description = generate_description_battery(new_result, sensor, dt_object)
        if description:
            current_battery_status = new_result.get("battery_status")
            if (prev_battery_status is None) or (prev_battery_status != current_battery_status):
                narrative_list.append((dt_object, description))
            prev_battery_status = current_battery_status
    elif sensor == "bluetooth": 
        description = generate_description_bluetooth(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
    elif sensor == "sensor_bluetooth": 
        description = generate_description_sensor_bluetooth(new_result, sensor, dt_object)
        if description:
            current_bluetooth_name = new_result.get("bt_name")
            if (prev_bluetooth_name is None) or (prev_bluetooth_name != current_bluetooth_name):
                narrative_list.append((dt_object, description))
            prev_bluetooth_name = current_bluetooth_name #save current result
    elif sensor == "calls": 
        description = generate_description_calls(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
    elif sensor == "installations": 
        description = generate_description_installations(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
    elif sensor == "keyboard": 
        description = generate_description_keyboard(new_result, sensor, dt_object)
        if description:
            if new_result["before_text"] == "":
                if prev_keyboard is not None:
                    narrative_list.append((dt_object, generate_description_keyboard(prev_keyboard, "keyboard", dt_object)))
            elif new_result["current_text"]  == "[]":
                narrative_list.append((dt_object, generate_description_keyboard(prev_keyboard, "keyboard", dt_object)))
            prev_keyboard = new_result #save current result
    elif sensor == "message": 
        description = generate_description_messages(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
    elif sensor == "screen": 
        description = generate_description_screen(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
    elif sensor == "touch": 
        description = generate_description_touch(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
    elif sensor == "wifi": 
        description = generate_description_wifi(new_result, sensor, dt_object)
        narrative_list.append((dt_object, description))
    elif sensor == "sensor_wifi": 
        narrative_list = generate_description_sensor_wifi(new_result, sensor, dt_object, prev_sensor_wifi, narrative_list)
        
# Generate map for location tracking
print("Locations: Generating Unimelb map...")
maps = []
for name, coordinates in MAP:
    def contains(lat, lon, poly=coordinates):
        return is_point_in_polygon(lat, lon, poly)
    result = {
        "name": name,
        "coordinates": coordinates,
        "contains": contains
    }
    maps.append(result)


# Generate distance matrix from new_results
print("Locations: Generate distance matrix...")
all_points = []
for new_result in new_results:
    # Convert timestamp to datetime object
    dt_object = datetime.fromtimestamp(int(new_result["timestamp"]) / 1000, timezone)
    # Check if required keys exist in the dictionary
    if "double_latitude" in new_result and "double_longitude" in new_result and "double_speed" in new_result:
        point = (
            float(new_result["double_latitude"]), # Latitude
            float(new_result["double_longitude"]), # Longitude
            dt_object, # Timestamp
            float(new_result["double_speed"]) # Speed
            )
        all_points.append(point)
        
# Convert the list of points into a NumPy array for easier processing
all_points = np.array(all_points)
print("Data size is ", len(all_points))

# Initialize distance matrix
# Compute the pairwise distances between all points using Haversine formula
# and store them in a symmetric matrix
distance_matrix = compute_distance_matrix_chunked(all_points)
distance_matrix = np.zeros((len(all_points), len(all_points)))
for i in range(len(all_points)):
    for j in range(i + 1, len(all_points)):
        distance = haversine(all_points[i, 0], all_points[i, 1], all_points[j, 0], all_points[j, 1])
        distance = distance.to_value() if isinstance(distance, u.Quantity) else distance
        distance_matrix[i, j] = float(distance)
        distance_matrix[j, i] = float(distance) # Mirror the distance for symmetry
print("Data size is ", len(all_points))




# Perform hierarchical clustering using the precomputed distance matrix
print("Locations: Performing hierarchical clustering")
try:
    if distance_matrix.any(): # Ensure the distance matrix is not empty
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=50,  # Define max distance threshold for clustering
            linkage=hierarchy_linkage,  # complete=longest, single=shortest (Use complete linkage (longest distance between clusters))
            affinity='precomputed'  # Provide precomputed distance matrix
        ).fit(distance_matrix)
        cluster_labels = clustering.labels_ # Get cluster labels
        n_clusters = len(set(cluster_labels)) # Count number of clusters
        print("Cluster size is ", str(n_clusters))
    else:
        raise ValueError("Distance matrix is empty, clustering cannot be performed")

    if all_points.any(): # Ensure there are points to process
        # Identify night-time data points (between 8 PM and 4 AM)
        night_labels = np.array([label for result, label in zip(new_results, cluster_labels) 
                                 if night_time_start <= datetime.fromtimestamp(result["timestamp"]/1000).hour or 
                                    datetime.fromtimestamp(result["timestamp"]/1000).hour <= night_time_end])
        if len(night_labels) == 0:
            raise ValueError("No data available for clustering")
        home_cluster_index = np.bincount(night_labels).argmax() # Identify home cluster
        
        cluster = []
        for cluster_id in set(cluster_labels):
            # Filter data by cluster ID, extract latitude and longitude for each cluster
            cluster_points = all_points[cluster_labels == cluster_id, :2].astype(float)  # Extract only latitude and longitude as floats
            # Skip if there is no data
            if len(cluster_points) == 0:
                continue
            cluster_center = np.mean(cluster_points, axis=0)  # Calculate the center of the cluster
            if cluster_id == home_cluster_index:  # Determine if the cluster is the home cluster
                place = "home"
                home_group_center = cluster_center  # Store the center of the home cluster in home_group_center
            else:
                included_places = is_point_in_any_polygon(cluster_center[0], cluster_center[1], maps)
                if included_places:
                    place = ", ".join(included_places)
                else:
                    place = "unknown"
            # Append cluster information to the list
            cluster.append((cluster_id, cluster_center[0], cluster_center[1], len(cluster_points), place))

        # Calculate distances of each cluster from home
        for i, cluster_entry in enumerate(cluster):
            cluster_id, center_lat, center_lon, num_points, place = cluster_entry
            distance_from_home = geodesic(home_group_center, (center_lat, center_lon)).meters
            cluster[i] = cluster_entry + (distance_from_home,)  # Update the tuple in the cluster list

    # Print cluster details
    if all_points:
        print("Cluster Centers:")
        for cluster_id, center_lat, center_lon, num_points, place, distance_from_home in cluster:
            print(f"Cluster {cluster_id}: Center Lat = {center_lat:.6f}, Center Lon = {center_lon:.6f}, N = {num_points}, Place = {place}, Distance = {distance_from_home:.3f}")

except NameError as e:
    print(f"Error: Variable not initialized - {e}")
except ValueError as e:
    print(f"Error: {e}")

# If a Google Maps API key is provided, perform reverse geocoding for unknown places
if GOOGLE_MAP_KEY:
    gmaps = googlemaps.Client(key=GOOGLE_MAP_KEY)
    try:
        for idx, (cluster_id, center_lat, center_lon, num_points, place, distance_from_home) in enumerate(cluster):
            if place == "unknown":
                reverse_geocode_data = gmaps.reverse_geocode((center_lat, center_lon))
                if reverse_geocode_data:
                    data = reverse_geocode_data[0]
                    formatted_address = data.get("formatted_address")
                    # if formatted_address.endswith(", Australia"):
                    #     formatted_address = formatted_address[:-11]  # Remove country: ", Australia"
                    #     if formatted_address.endswith(common_addresses):
                    #         formatted_address = formatted_address[:-9]  # If it's a common address, remove " VIC XXXX"

                    place_name = None
                    place_type = None

                    # Search within address_components
                    for component in data.get("address_components", []):
                        if isinstance(component, dict) and component.get("types"):
                            place_type = component["types"][0]
                            place_name = f"{component.get('long_name', '')},"
                            break

                    # Ensure place_type, place_name, and formatted_address are not None, use empty strings if they are
                    place_type = place_type or ""
                    place_name = place_name or ""
                    formatted_address = formatted_address or ""

                    place = place_type + place_name + formatted_address

                cluster[idx] = (cluster_id, center_lat, center_lon, num_points, place, distance_from_home)
    except Exception as e:
        print(f"Error in Google Maps API request: {e}")

# Display the updated cluster information
if all_points:
    print("Cluster Centers (Updated):")
    for cluster_id, center_lat, center_lon, num_points, place, distance_from_home in cluster:
        print(f"Cluster {cluster_id}: Center Lat = {center_lat:.6f}, Center Lon = {center_lon:.6f}, N = {num_points}, Place = {place}, Distance = {distance_from_home:.3f}")

        
        
# Generate descriptions for locations based on clustering results
# Contents of all_points: 0: result["double_latitude"], 1: result["double_longitude"], 2: dt_object.strftime("%c"), 3: result["double_speed"]
# Contents of cluster: 0: cluster_id, 1: center_lat, 2: center_lon, 3: num_points, 4: place, 5: distance_from_home
if all_points:
    narrative_list = generate_description_locations(narrative_list, all_points, cluster_labels, cluster)

# Write output description to a text file
print("Output description into text file...")
narrative_list = sorted(set(narrative_list))
text = "\n".join([str(item[1]) for item in narrative_list])

# Remove system UI entries if required
if DISCARD_SYSTEM_UI:
    cleaned_narrative_list = [x for x in narrative_list if "System UI" not in x[1]]
    text = "\n".join([str(item[1]) for item in cleaned_narrative_list])
    
# Save to file
with open(output_file, 'w') as file:
    file.write(text)
