from dorna2 import Dorna
import time

CYCLES = 20

def build_sequence(cycles):

    cmd_list = []

    for _ in range(cycles):

        # A → B → C → D
        cmd_list.append('{"cmd":"lmove","rel":0,"x":386.777475,"y":-104.275356,"z":169.734268,"a":176.101107,"b":-33.250912,"c":6.131734,"vel":100,"acc":800,"jerk":1000,"cont":1,"corner":100}')
        cmd_list.append('{"cmd":"lmove","rel":0,"x":266.843852,"y":-104.206306,"z":169.708871,"a":176.111387,"b":-33.251148,"c":6.142103,"vel":100,"acc":800,"jerk":1000,"cont":1,"corner":100}')
        cmd_list.append('{"cmd":"lmove","rel":0,"x":266.711805,"y":175.773619,"z":169.707272,"a":176.107832,"b":-33.244479,"c":6.112397,"vel":100,"acc":800,"jerk":1000,"cont":1,"corner":100}')
        cmd_list.append('{"cmd":"lmove","rel":0,"x":386.783937,"y":175.752657,"z":169.835723,"a":176.116134,"b":-33.260301,"c":6.145552,"vel":100,"acc":800,"jerk":1000,"cont":1,"corner":100}')

        # D → C → B
        cmd_list.append('{"cmd":"lmove","rel":0,"x":266.711805,"y":175.773619,"z":169.707272,"a":176.107832,"b":-33.244479,"c":6.112397,"vel":100,"acc":800,"jerk":1000,"cont":1,"corner":100}')
        cmd_list.append('{"cmd":"lmove","rel":0,"x":266.843852,"y":-104.206306,"z":169.708871,"a":176.111387,"b":-33.251148,"c":6.142103,"vel":100,"acc":800,"jerk":1000,"cont":1,"corner":100}')

    return cmd_list


def main(robot):

    robot.play(-1, {"cmd": "motor", "motor": 1})
    time.sleep(1)

    print("Building full sequence...")

    sequence = build_sequence(CYCLES)

    print("Sending full sequence...")

    # 🔥 THIS WORKS in Dorna2
    for cmd in sequence:
        robot.play(-1, eval(cmd))

    print("Sequence sent")

    # wait for completion
    while True:
        stat = robot.play(-1, {"cmd": "stat"})
        if stat["stat"] == 0:
            break
        time.sleep(0.05)

    print("Done")


if __name__ == "__main__":
    robot = Dorna()
    try:
        robot.connect(host="192.168.1.24", port=443)
        time.sleep(1)

        main(robot)

    finally:
        robot.close()