import time
import os


def remove_untested_combinations():
    path = "/Users/alexdunn/my-env/TurboWorks/Perovskite_task/untested_combinations.pickle"
    if os.path.isfile(path):
        os.remove(path)
        print "file removed"
    else:
        print "No file generated yet"
    time.sleep(1800)



if __name__ == "__main__":
    for i in range(10000):
        remove_untested_combinations()
