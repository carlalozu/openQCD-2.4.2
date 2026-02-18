import re
import csv
import sys

# Patterns for the header info
re_lattice = re.compile(r'(?P<lat>\d+x\d+x\d+x\d+) lattice')
re_local   = re.compile(r'(?P<loc>\d+x\d+x\d+x\d+) thread-local lattice')
re_threads = re.compile(r'(?P<thr>\d+) OpenMP thread')

# Patterns for the Volume section (Key: Value)
re_kv = re.compile(r'(?P<key>[\w\s&()/_]+):\s+(?P<val>[\d\.e\+-]+)')

writer = csv.writer(sys.stdout)
writer.writerow([
    "lattice", "local_lattice", "threads", "volume", 
    "vol_per_thread", "repetitions", "avg_time_sec", 
    "total_mflops", "time_per_point_sec", "mflops_per_thread", "result"
])

# State variables to hold data across lines
data = {}

for line in sys.stdin:
    line = line.strip()
    if not line: continue

    # Capture header info
    m_lat = re_lattice.search(line)
    if m_lat: data['lattice'] = m_lat.group('lat')
    
    m_loc = re_local.search(line)
    if m_loc: data['local_lattice'] = m_loc.group('loc')
    
    m_thr = re_threads.search(line)
    if m_thr: data['threads'] = m_thr.group('thr')

    # Capture key-value metrics
    m_kv = re_kv.search(line)
    if m_kv:
        key, val = m_kv.group('key'), m_kv.group('val')
        
        if "Volume per thread" in key: data['vpt'] = val
        elif "Volume" in key:           data['vol'] = val
        elif "repetitions" in key:     data['rep'] = val
        elif "Average time" in key:    data['avg'] = val
        elif "Total performance" in key: data['perf'] = val
        elif "Time per lattice" in key: data['tpl'] = val
        elif "Performance per" in key:  data['ppt'] = val
        elif "Result" in key:          data['res'] = val

        # "Result" is the last line of the block; trigger the row write
        if "Result" in key:
            writer.writerow([
                data.get('lattice'), data.get('local_lattice'), data.get('threads'),
                data.get('vol'), data.get('vpt'), data.get('rep'),
                data.get('avg'), data.get('perf'), data.get('tpl'),
                data.get('ppt'), data.get('res')
            ])
            # Optional: data.clear() if processing multiple concatenated logs
            