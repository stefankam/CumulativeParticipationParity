import re 
import json

class AvailabilityTrace:
    def __init__(self, trace_path, device_index):
        self.device_key = str(device_index)  # keys are "0", "1", etc. in JSON
        self.trace = self._load_device_trace(trace_path)
        self.pointer = 0
        self.battery_on = False
        self.wifi_on = False

    def _load_device_trace(self, path):
        with open(path, 'r') as f:
            data = json.load(f)

        if self.device_key not in data:
            raise ValueError(f"Device {self.device_key} not found in trace file")

        # "messages" is one long string, split into individual events
        messages_str = data[self.device_key]["messages"]
        events = [line.split("\t")[-1].strip() for line in messages_str.splitlines() if line.strip()]
        return events

    def is_available(self):
        if not self.trace:
            return False

        if self.pointer >= len(self.trace):
            self.pointer = 0  # loop over trace

        event = self.trace[self.pointer]
        self.pointer += 1

        # Update state based on event
        if event == "battery_charged_on":
            self.battery_on = True
        elif event == "battery_charged_off":
            self.battery_on = False
        elif event == "wifi_on":
            self.wifi_on = True
        elif event == "wifi_off":
            self.wifi_on = False

        print(f"[Trace] Event: {event}, Battery: {self.battery_on}, WiFi: {self.wifi_on}")
        return self.battery_on and self.wifi_on

    def advance(self):
        if self.pointer < len(self.trace) - 1:
            self.pointer += 1
        print(f"[Trace] Advanced to pointer: {self.pointer}, Available: {self.trace[self.pointer]}")


    @staticmethod
    def extract_device_index(device_id_str):
        match = re.search(r'(\d+)$', device_id_str)
        return int(match.group(1)) if match else 0


def load_availability_traces(path):
    """
    Read availability traces from a JSON file.
    JSON format expected:
    {
        "0": {"messages": "wifi_on\nbattery_charged_on\nwifi_off\n..."},
        "1": {"messages": "..."}
    }
    Returns a dict keyed by "h1", "h2", ... with list of events.
    """
    traces = {}
    with open(path, "r") as f:
        data = json.load(f)

    for device_index, device_data in data.items():
        host_name = f"h{int(device_index)+1}"
        messages_str = device_data["messages"]
        events = [line.strip() for line in messages_str.splitlines() if line.strip()]
        traces[host_name] = events

    return traces

def extract_availability_vectors(path):
   traces = load_availability_traces(path)
   def extract_vector(trace, length=100):
      wifi, charging = False, False
      vector = []
      for event in trace:
         if "wifi" in event:
            wifi = "off" not in event
         elif "battery_charged" in event:
            charging = "off" not in event
         # After each relevant event, record availability status
         availability = wifi and charging
         vector.append(int(availability))
      return vector[:length] + [0] * max(0, length - len(vector))

   availability_vectors = {
      device: extract_vector(trace) for device, trace in traces.items()
   }

   #print("availability_vectors: ", availability_vectors)
   return availability_vectors
