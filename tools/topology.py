import gzip
import re


def read_prmtop(path, wanted_flags):
    sections = {name: [] for name in wanted_flags}
    current = None
    width = None
    value_type = None

    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="ascii") as topology_file:
        for raw_line in topology_file:
            line = raw_line.rstrip("\n")
            if line.startswith("%FLAG "):
                name = line.split(maxsplit=1)[1]
                current = name if name in wanted_flags else None
                continue
            if line.startswith("%FORMAT"):
                if current is not None:
                    match = re.fullmatch(
                        r"%FORMAT\(\d+([A-Za-z])(\d+)(?:\.\d+)?\)", line
                    )
                    value_type = match.group(1).upper()
                    width = int(match.group(2))
                continue
            if current is None:
                continue

            for start in range(0, len(line), width):
                token = line[start : start + width].strip()
                if not token:
                    continue
                if value_type == "I":
                    value = int(token)
                elif value_type in {"E", "F", "D"}:
                    value = float(token.replace("D", "E"))
                else:
                    value = token
                sections[current].append(value)

    return sections
