import json
import sys

json_path = sys.argv[1]
seed_offset = int(sys.argv[2])

#durations = []

with open(json_path) as f:
	durations = [json.loads(line.strip())["duration"] 
	            for line in f]
	assert len(durations) > seed_offset
	durations = durations[seed_offset:]
	print(sum(durations))