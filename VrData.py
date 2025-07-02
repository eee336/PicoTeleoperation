import numpy as np
import socket
import json

class VrData:
    def __init__(self):
        self.data = {}
        self.UDP_IP="127.0.0.1"
        self.UDP_PORT = 5006
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.UDP_IP, self.UDP_PORT))

        self.position=np.array([0,0,0])
        self.quat=np.array([0,0,0,1])

        self.button_active= 0

    def receive_data(self):
        data, _ = self.sock.recvfrom(1024)
        data = json.loads(data.decode())
        self.data=data
        # 位置
        pos = self.data["info"]["right"]["position"]
        self.position=[pos['x'],pos['y'],pos['z']]

        #姿势
        rot = self.data["info"]["right"]["orientation"]
        self.quat = [rot['x'],rot['y'],rot['z'],rot['w']]


        # 按钮是否出触发
        self.active = data["buttons"]["right"]["select"]

        return self.position, self.quat, self.button_active
