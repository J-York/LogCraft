from logafe.parsing import parse
from logafe.clustering import cluster
from logafe.sampling import sampling_sequential

def preprocess(logname,format_index):
    parse(logname,format_index)
    cluster(logname)
    sampling_sequential(log_name=logname)


if __name__ == "__main__":
    logname = "BGL"
    # logfile = "BGL.log"
    preprocess(logname)