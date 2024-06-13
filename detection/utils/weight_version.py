import re


def version_to_tuple(version):
    return tuple(map(int, version.split('.')))


def is_valid_version_format(version):
    return bool(re.match(r'^\d+\.\d+\.\d+$', version))


def is_version_in_range(version, version1, version2):
    if not is_valid_version_format(version):
        return False

    v = version_to_tuple(version)
    v1 = version_to_tuple(version1)
    v2 = version_to_tuple(version2)

    if v1 > v2:
        v1, v2 = v2, v1

    return v1 <= v <= v2