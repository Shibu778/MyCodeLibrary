def extract_mag_data(outcar_path="./OUTCAR", mag_data_path="./mag_data.txt"):
    import os

    command = (
        'grep "magnetization (x)" '
        + outcar_path
        + " -A 32 | tail -n 32 | tee "
        + mag_data_path
    )
    os.system(command)
    return 0


def read_mag_data(filename):
    """
    Read the mag_data file extracted from OUTCAR using the following command.

    `grep "magnetization (x)" OUTCAR -A 32 | tail -n 32 | tee mag_data.txt`

    """
    with open(filename, "r") as f:
        data = f.readlines()

    # Split between spaces
    data = [d.split() for d in data]

    # Remove the first empty line
    data = data[1:-1]

    keys = ["atom_index"]
    for key in data[0][3:]:
        keys.append(str(key))

    mag_data = {}
    mag_data["keys"] = keys
    mag_data["tot"] = [float(d) for d in data[-1][1:]]

    # Delete lines that are not required
    del data[0]
    del data[0]
    del data[-2]
    del data[-1]

    mag_data["individual"] = []

    for d in data:
        tmp = []
        tmp.append(int(d[0]))
        tmp += [float(s) for s in d[1:]]
        mag_data["individual"].append(tmp)

    return mag_data


if __name__ == "__main__":
    filename = "./mag_data.txt"
    outcar_path = "./OUTCAR"
    # To use the following function `grep` must be working in your system
    extract_mag_data(outcar_path=outcar_path)
    mag_data = read_mag_data(filename)
    print(mag_data)
