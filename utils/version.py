__version__ = "1.1.3"
last_acceptable_version = "1.1.1"
def get_version():
    version_split = __version__.split(".")
    spec_version = (
        (10000 * int(version_split[0]))
        + (100 * int(version_split[1]))
        + (1 * int(version_split[2]))
    )
    
    return spec_version