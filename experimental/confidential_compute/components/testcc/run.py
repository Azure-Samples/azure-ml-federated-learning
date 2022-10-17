import os
import sys
import socket

local_hostname = socket.gethostname()
local_ip = socket.gethostbyname(local_hostname)

print("Hello World from " + local_hostname + " (" + local_ip + ")")

for key in os.environ:
    print("ENV: {}={}".format(key, os.environ[key]))
