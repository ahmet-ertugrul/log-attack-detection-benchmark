"""Download the UWF-ZeekData22 and UWF-ZeekData24 CSV files used in the experiments and verify their SHA-256 checksums.

The datasets are published by the University of West Florida under CC BY 4.0 (https://datasets.uwf.edu/).
Files are saved under data/ with the short names expected by experiments.py. Total size is about 470 MB.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://datasets.uwf.edu/data"
DATA = Path(__file__).resolve().parent / "data"

FILES = {
    # local name: (remote path, sha256)
    "UWF-ZeekData22/part1.csv": ("UWF-ZeekData22/csv/part-00000-0af89d10-df53-44fd-b124-a8a496fd5023-c000.csv",
                                 "a7c005001b2242fde365902fc6d14f98b0a298431a6b85884a70c2176ab8b742"),
    "UWF-ZeekData22/part2.csv": ("UWF-ZeekData22/csv/part-00000-15e3dd03-ea76-429e-a52a-ce96a90517f9-c000.csv",
                                 "991d7cd72f299484ab2892df21a02cd838fea0dce3e92eab6a406c69f78e79c2"),
    "UWF-ZeekData22/part3.csv": ("UWF-ZeekData22/csv/part-00000-318611a1-7cdc-4dd0-9348-c6368917fd0c-c000.csv",
                                 "7c5b8f344db31c5b4a8a3fbbe647aeb146c1242409994f5d0e34b2cd058866f1"),
    "UWF-ZeekData22/part4.csv": ("UWF-ZeekData22/csv/part-00000-5b4f5c3f-e8a9-4020-8fa1-e8985f7c27f3-c000.csv",
                                 "b0779cfe6e6d086ca6c7fa0500775e839187e4d18059016dfcd53e98efe37521"),
    "UWF-ZeekData22/part5.csv": ("UWF-ZeekData22/csv/part-00000-95e0a460-e7c5-4b35-8367-c2e6fbbcf9e1-c000.csv",
                                 "2d422f1bc86dc9e9fc5658a0ad07cd3b3c62097c3eed236577d62260678caa7c"),
    "UWF-ZeekData24/Benign.csv": ("UWF-ZeekData24/csv/Benign/part-00000-b039213f-1208-4f1b-b71d-5e8e3a4a2939-c000.csv",
                                  "272677c8c7bbe7ab57b040f63ed8bbf92d0b5716adb05444258274eed000ab3f"),
    "UWF-ZeekData24/Reconnaissance.csv": ("UWF-ZeekData24/csv/Reconnaissance/part-00000-49ed1cc6-3205-4c76-8e28-4da9abf78363-c000.csv",
                                          "13bf1c7fe9364a91a18d858683fee999baf6567ca65e2077829df089107ff7b0"),
    "UWF-ZeekData24/Credential_Access.csv": ("UWF-ZeekData24/csv/Credential_Access/part-00000-912fdc44-5727-4d42-8fd6-a0e206ba29f8-c000.csv",
                                             "6448eeebbb5dfaf88bb45f81a2fa8fd3b2a3e635090da2e50b0177bd71430424"),
    "UWF-ZeekData24/Initial_Access.csv": ("UWF-ZeekData24/csv/Initial_Access/part-00000-9a37b839-429e-444b-82a5-a6d5e69dad7e-c000.csv",
                                          "faf6b9e07a7dab23ce364898772af062f51ddef4feb9329778750f8ed699d0e6"),
    "UWF-ZeekData24/Defense_Evasion.csv": ("UWF-ZeekData24/csv/Defense_Evasion/part-00000-de4985c7-284a-4b6e-b507-e69a2d172582-c000.csv",
                                           "801594d09ce3b9a18f86199a4dbff04511e375910f3d0d932a6dcde1153cb101"),
    "UWF-ZeekData24/Exfiltration.csv": ("UWF-ZeekData24/csv/Exfiltration/part-00000-6a530c25-0f6b-46a1-ba16-c6b658ef75e8-c000.csv",
                                        "b485153727b8a630582838eebf75c924cb0b49476b69de1b728b85d4371827f0"),
    "UWF-ZeekData24/Persistence.csv": ("UWF-ZeekData24/csv/Persistence/part-00000-fb5a764a-e65e-4e10-b6fb-1a189589d5e0-c000.csv",
                                       "52b19554a53a7da044fbf25f8546702e00043cca43673ebe41ee091cfb2009c8"),
    "UWF-ZeekData24/Privilege_Escalation.csv": ("UWF-ZeekData24/csv/Privilege_Escalation/part-00000-d8fcf83a-aed1-4cb1-babf-581208088293-c000.csv",
                                                "5933fa87bf55f54821705754be357ea1fa0e347b440c3af0590be6e03bd04877"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    failed = []
    for local, (remote, digest) in FILES.items():
        dst = DATA / local
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and sha256(dst) == digest:
            print(f"ok (cached)  {local}")
            continue
        print(f"downloading  {local}")
        urllib.request.urlretrieve(f"{BASE}/{remote}", dst)
        if sha256(dst) == digest:
            print(f"ok           {local}")
        else:
            print(f"CHECKSUM MISMATCH  {local}")
            failed.append(local)
    if failed:
        print("\nThe following files differ from the versions used in the chapter:", *failed, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
