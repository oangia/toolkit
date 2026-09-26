import time

def log(msg):
    if debug==False:
        return
    print(time.time()-start)
    print(msg)
