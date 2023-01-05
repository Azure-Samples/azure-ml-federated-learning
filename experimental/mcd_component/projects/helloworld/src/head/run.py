import os

print("Hello head")

print("MCD_RANK={}".format(os.environ["MCD_RANK"]))
print("MCD_SIZE={}".format(os.environ["MCD_SIZE"]))
print("MCD_RUN_ID={}".format(os.environ["MCD_RUN_ID"]))
print("MCD_HEAD={}".format(os.environ["MCD_HEAD"]))
print("MCD_WORKERS={}".format(os.environ["MCD_WORKERS"]))

import pythonping

print("Pinging worker nodes")

for worker_ip in os.environ["MCD_WORKERS"].split(","):
    print("Worker node address: {}".format(worker_ip))
    response_list = pythonping.ping(worker_ip, count=100)
    for response in response_list:
        print(response)
