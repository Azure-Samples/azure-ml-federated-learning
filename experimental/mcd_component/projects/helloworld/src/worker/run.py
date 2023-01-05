import os

print("Hello head")

print("MCD_RANK={}".format(os.environ["MCD_RANK"]))
print("MCD_SIZE={}".format(os.environ["MCD_SIZE"]))
print("MCD_RUN_ID={}".format(os.environ["MCD_RUN_ID"]))
print("MCD_HEAD={}".format(os.environ["MCD_HEAD"]))
print("MCD_WORKERS={}".format(os.environ["MCD_WORKERS"]))

import pythonping

print("Pinging head node")
print("Head node address: {}".format(os.environ["MCD_HEAD"]))
response_list = pythonping.ping(os.environ["MCD_HEAD"], count=5)
for response in response_list:
    print(response)

print("Head node hostname: head")
response_list = pythonping.ping("head", count=5)
for response in response_list:
    print(response)
